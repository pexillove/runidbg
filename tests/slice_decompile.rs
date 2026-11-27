//! 动态污点切片 → 单路径符号提升 → 代数/MBA 化简 → 自校验精简伪 C（与 SMT-LIB2）端到端集成测试
//!
//! 直接用 `good64` 现场汇编真实 ARM64 算法函数（含死代码干扰、栈 spill、ROTR、BSWAP、S-Box 查表、
//! OLLVM 风格 MBA 混淆），在 Unicorn 中以 `TraceLevel::Full` 真实执行并落盘 `.idx`/`.ext`，
//! 随后调用 `TraceReader::decompile_slice`、QuickJS `tr.taintToC` 与 MCP `trace_decompile`，
//! 并在宿主机存在 `cc` 编译器时将生成的伪 C 真编译成可执行程序做跨输入交叉校验！

use std::fs;
use std::process::Command;

use runidbg_arm64_emu_tracer::good64::assemble;
use runidbg_arm64_emu_tracer::tracer::{
    BinaryTracer, DecompileOptions, TaintSeed, TraceLevel, TraceReader,
};
use runidbg_arm64_emu_tracer::{AndroidEmulator, CpuBackend};

fn unique_temp_prefix(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("runidbg_decomp_{tag}_{}_{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// 辅助：在 Unicorn 中以 Full Trace 执行一段 `good64` 汇编代码，返回 `(idx_path, ext_path, out_u32)`
fn trace_good64_snippet(
    tag: &str,
    asm_src: &str,
    in_bytes: &[u8],
    sbox_bytes: &[u8],
) -> (std::path::PathBuf, std::path::PathBuf, u64, u64, u64, u32) {
    let dir = unique_temp_prefix(tag);
    let idx_path = dir.join("trace.idx");
    let ext_path = dir.join("trace.ext");

    let mut emu = AndroidEmulator::builder().build().expect("build emu");
    let code_base = 0x4000_0000u64;
    let data_base = 0x4001_0000u64;
    let in_buf = data_base;
    let sbox_buf = data_base + 0x100;
    let out_buf = data_base + 0x200;

    emu.cpu_mut()
        .mem_map(code_base, 0x20000, unicorn_engine::unicorn_const::Prot::ALL)
        .expect("map code+data");
    let code = assemble(asm_src, code_base).expect("good64 assemble");
    emu.cpu_mut().mem_write(code_base, &code).unwrap();
    emu.cpu_mut().mem_write(in_buf, in_bytes).unwrap();
    if !sbox_bytes.is_empty() {
        emu.cpu_mut().mem_write(sbox_buf, sbox_bytes).unwrap();
    }

    let tracer =
        BinaryTracer::new_dual_mmap(&idx_path, &ext_path, 4096, 16384, TraceLevel::Full)
            .expect("create BinaryTracer");
    emu.set_tracer(tracer).unwrap();
    emu.attach_tracer(code_base, code_base + 0x1000).unwrap();

    let _ = emu
        .call_address(code_base, &[in_buf, sbox_buf, out_buf])
        .expect("call snippet");

    emu.tracer().flush().unwrap();

    let mut out_raw = [0u8; 4];
    emu.cpu_mut().mem_read(out_buf, &mut out_raw).unwrap();
    (
        idx_path,
        ext_path,
        in_buf,
        sbox_buf,
        out_buf,
        u32::from_le_bytes(out_raw),
    )
}

#[test]
fn test_slice_decompile_crypto_rotr_bswap_sbox_and_spill_elimination() {
    // 输入 4 字节小端 u32 = 0x11223344
    let input_u32 = 0x1122_3344u32;
    let sbox: [u8; 16] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    ];

    // 真实 ARM64 汇编：混入大量死代码、栈 spill、手写 ROTR32、S-Box 查表与 BSWAP32
    let asm = r#"
        // 干扰死代码（不应出现在精简伪 C 中）
        mov x11, #0xdead
        mov x12, #0xbeef
        eor x13, x11, x12
        mul x14, x13, x11

        // 读取输入 32-bit 字 -> w4
        ldr w4, [x0]
        // 常量异或 0x5a
        mov w10, #0x5a
        eor w4, w4, w10
        // 栈 spill：写入栈再读出到 w7，应被字节模型零损耗消除！
        str w4, [sp, #-16]!
        add x15, x14, #1
        ldr w7, [sp], #16

        // 手写 32-bit 循环右移 7 位：(w7 >> 7) | (w7 << 25)
        lsr w8, w7, #7
        lsl w9, w7, #25
        orr w7, w8, w9

        // S-Box 查表：sbox[w7 & 0xf]
        and w5, w7, #0xf
        ldrb w6, [x1, x5]

        // 字节序反转 + 累加 S-Box 查表值，写出到 [x2]
        rev w7, w7
        add w7, w7, w6
        str w7, [x2]
        mov w0, w7
        ret
    "#;

    let (idx_path, ext_path, _in_buf, sbox_buf, out_buf, actual_out) =
        trace_good64_snippet("crypto", asm, &input_u32.to_le_bytes(), &sbox);

    // 独立数学预期值校验
    let expected = {
        let v = (input_u32 ^ 0x5a).rotate_right(7);
        let sb = u32::from(sbox[(v & 0xf) as usize]);
        v.swap_bytes().wrapping_add(sb)
    };
    assert_eq!(actual_out, expected);

    let reader = TraceReader::open(&idx_path, Some(&ext_path)).unwrap();
    let report = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: out_buf,
            len: 4,
        },
        &DecompileOptions {
            function_name: "crypto_round".to_string(),
            emit_smt2: true,
            ..Default::default()
        },
    );

    assert_eq!(
        report.verified,
        Some(true),
        "自校验必须逐 bit 通过:\n{}",
        report.format_summary()
    );
    assert_eq!(report.expected_output, Some(u64::from(expected)));
    assert_eq!(report.evaluated_output, Some(u64::from(expected)));
    assert!(
        report.c_code.contains("ROTR32("),
        "应自动识别 ROTR32:\n{}",
        report.c_code
    );
    assert!(
        report.c_code.contains("BSWAP32("),
        "应自动识别 BSWAP32:\n{}",
        report.c_code
    );
    assert!(
        report.c_code.contains("sbox_0["),
        "应自动识别 S-Box 查表:\n{}",
        report.c_code
    );
    assert!(
        !report.c_code.contains("0xdead") && !report.c_code.contains("0xbeef"),
        "死代码常量不应泄露到精简伪 C:\n{}",
        report.c_code
    );
    assert_eq!(report.tables.len(), 1);
    assert_eq!(report.tables[0].base_addr, sbox_buf);

    let smt = report.smt2.as_deref().expect("expected smt2 output");
    assert!(smt.contains("(set-logic QF_UFBV)"));
    assert!(smt.contains("(declare-fun sbox_0"));
    assert!(smt.contains("(check-sat)"));
}

#[test]
fn test_slice_decompile_mba_deobfuscation_and_real_c_compilation() {
    // 构造输入缓冲区 [a=0x12345678, b=0x9abcdef0]
    let a = 0x1234_5678u32;
    let b = 0x9ABC_DEF0u32;
    let mut in_bytes = Vec::new();
    in_bytes.extend_from_slice(&a.to_le_bytes());
    in_bytes.extend_from_slice(&b.to_le_bytes());

    // OLLVM 风格 MBA 混淆：
    //   t1 = (a ^ b) + ((a & b) << 1)   ==> 等价于 a + b
    //   t2 = (t1 | 0x1337) - (t1 & 0x1337) ==> 等价于 t1 ^ 0x1337
    let asm = r#"
        ldr w4, [x0]
        ldr w5, [x0, #4]
        eor w6, w4, w5
        and w7, w4, w5
        lsl w7, w7, #1
        add w8, w6, w7          // MBA(a + b)
        mov w9, #0x1337
        orr w10, w8, w9
        and w11, w8, w9
        sub w12, w10, w11       // MBA(t1 ^ 0x1337)
        str w12, [x2]
        ret
    "#;

    let (idx_path, ext_path, _in_buf, _sbox_buf, out_buf, actual_out) =
        trace_good64_snippet("mba", asm, &in_bytes, &[]);

    let expected = a.wrapping_add(b) ^ 0x1337;
    assert_eq!(actual_out, expected);

    let reader = TraceReader::open(&idx_path, Some(&ext_path)).unwrap();
    let report = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: out_buf,
            len: 4,
        },
        &DecompileOptions {
            function_name: "mba_demystified".to_string(),
            emit_smt2: true,
            ..Default::default()
        },
    );

    assert_eq!(
        report.verified,
        Some(true),
        "MBA 化简结果自校验必须通过:\n{}",
        report.format_summary()
    );
    assert!(
        report.simplified_nodes < report.raw_nodes,
        "MBA 化简应显著缩减 DAG 节点数 (raw={}, simplified={})",
        report.raw_nodes,
        report.simplified_nodes
    );
    assert!(
        report.c_code.contains("0x1337"),
        "化简后应保留关键异或常量:\n{}",
        report.c_code
    );

    // 终极验证：如果宿主机安装了 `cc`，将生成的伪 C 真编译成原生二进制，并用【全新未见过的输入】交叉验证！
    if Command::new("cc").arg("--version").output().is_ok() {
        let dir = idx_path.parent().unwrap();
        let c_file = dir.join("verify_mba.c");
        let bin_file = dir.join("verify_mba");
        let harness = format!(
            r#"
{c_code}
#include <stdio.h>
int main(void) {{
    uint8_t buf1[8] = {{ 0x78, 0x56, 0x34, 0x12, 0xf0, 0xde, 0xbc, 0x9a }};
    uint32_t r1 = mba_demystified(buf1);
    if (r1 != 0x{expected:08x}u) {{
        fprintf(stderr, "r1 mismatch: 0x%x vs 0x{expected:08x}\n", r1);
        return 1;
    }}
    // 全新未见输入：a2=0xCAFEBABE, b2=0x01020304 -> (a2 + b2) ^ 0x1337
    uint8_t buf2[8] = {{ 0xbe, 0xba, 0xfe, 0xca, 0x04, 0x03, 0x02, 0x01 }};
    uint32_t exp2 = (0xCAFEBABEu + 0x01020304u) ^ 0x1337u;
    uint32_t r2 = mba_demystified(buf2);
    if (r2 != exp2) {{
        fprintf(stderr, "r2 mismatch: 0x%x vs 0x%x\n", r2, exp2);
        return 2;
    }}
    return 0;
}}
"#,
            c_code = report.c_code
        );
        fs::write(&c_file, harness).unwrap();
        let compile = Command::new("cc")
            .args(["-O2", "-Wall", "-Werror"])
            .arg(&c_file)
            .arg("-o")
            .arg(&bin_file)
            .output()
            .expect("invoke cc");
        assert!(
            compile.status.success(),
            "生成的伪 C 应以 -Wall -Werror 零告警通过编译!\nstderr: {}\nC:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            report.c_code
        );
        let run = Command::new(&bin_file).output().expect("run verify_mba");
        assert!(
            run.status.success(),
            "编译后的伪 C 在跨输入测试上运行失败: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

#[cfg(all(feature = "js", feature = "mcp"))]
#[test]
fn test_slice_decompile_js_and_mcp_end_to_end() {
    use runidbg_arm64_emu_tracer::script::mcp::McpServer;
    use runidbg_arm64_emu_tracer::script::JsSession;
    use serde_json::json;
    use std::path::Path;

    let input_u32 = 0x55AA_1234u32;
    let asm = r#"
        ldr w4, [x0]
        mov w10, #0x33
        eor w4, w4, w10
        add w4, w4, #0x100
        str w4, [x2]
        ret
    "#;
    let (idx_path, ext_path, _in_buf, _sbox_buf, out_buf, actual_out) =
        trace_good64_snippet("js_mcp", asm, &input_u32.to_le_bytes(), &[]);

    // 1. QuickJS `tr.taintToC(...)`
    let js = format!(
        r#"
        (() => {{
            const tr = Trace.open({idx:?}, {ext:?});
            const rep = tr.taintToC({{
                addr: {out_buf},
                len: 4,
                functionName: "js_slice_fn",
                emitSmt2: true,
            }});
            tr.close();
            if (rep.verified !== true) throw new Error("JS verified=" + rep.verified);
            if (!rep.cCode.includes("js_slice_fn")) throw new Error("missing fn name");
            if (!rep.smt2 || !rep.smt2.includes("QF_BV")) throw new Error("missing smt2");
            return rep;
        }})()
        "#,
        idx = idx_path.to_string_lossy(),
        ext = ext_path.to_string_lossy(),
    );
    let session = JsSession::new(&[], Path::new("."), true).unwrap();
    let js_res = session.eval_json_string(&js).unwrap();
    assert!(js_res.contains(&format!("0x{actual_out:x}")));

    // 2. MCP `trace_decompile`
    let mut srv = McpServer::new(Path::new("."), None).unwrap();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {
            "name": "trace_decompile",
            "arguments": {
                "idxPath": idx_path.to_string_lossy(),
                "extPath": ext_path.to_string_lossy(),
                "addr": format!("0x{out_buf:x}"),
                "len": 4,
                "functionName": "mcp_slice_fn",
                "emitSmt2": true
            }
        }
    });
    let resp: serde_json::Value =
        serde_json::from_str(&srv.handle_message(&req.to_string()).unwrap()).unwrap();
    let payload: serde_json::Value =
        serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(payload["verified"], true);
    assert!(payload["cCode"].as_str().unwrap().contains("mcp_slice_fn"));
    assert!(payload["smt2"].as_str().unwrap().contains("QF_BV"));
}

#[test]
fn test_slice_decompile_audit_regressions() {
    // 1. P0-1 & P1-2: 含未建模指令 cls (Opaque) 绝不可谎报 verified == Some(true)，
    //    且 Opaque 不产生嵌套注释、恶意 functionName 被安全净化并通过 cc -Wall -Werror 语法检查
    let in_u64 = 0x1122_3344_5566_7788u64;
    let asm_opaque = r#"
        ldr x4, [x0]
        .word 0xdac01486
        add x6, x6, #1
        str x6, [x2]
        ret
    "#;
    let (idx_o, ext_o, _, _, out_o, _) =
        trace_good64_snippet("audit_opaque", asm_opaque, &in_u64.to_le_bytes(), &[]);
    let reader_o = TraceReader::open(&idx_o, Some(&ext_o)).unwrap();
    let rep_o = reader_o.decompile_slice(
        &TaintSeed::Mem {
            addr: out_o,
            len: 8,
        },
        &DecompileOptions {
            function_name: "evil(){return 0;} int bad".to_string(),
            emit_smt2: true,
            ..Default::default()
        },
    );
    assert_eq!(
        rep_o.verified, None,
        "含 Opaque 指令绝不可假阳性 verified == Some(true)"
    );
    assert!(
        !rep_o.c_code.contains("evil()"),
        "恶意 functionName 必须被安全净化:\n{}",
        rep_o.c_code
    );
    assert!(
        rep_o.smt2.as_deref().unwrap().contains("opaque_"),
        "Opaque 在 SMT2 中应导出为未约束符号常量"
    );
    if Command::new("cc").arg("--version").output().is_ok() {
        let c_file = idx_o.parent().unwrap().join("opaque_syntax.c");
        fs::write(&c_file, &rep_o.c_code).unwrap();
        let st = Command::new("cc")
            .args(["-fsyntax-only", "-Wall", "-Werror"])
            .arg(&c_file)
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "含 Opaque 的生成 C 必须无语法错误（无嵌套注释）:\n{}\nC:\n{}",
            String::from_utf8_lossy(&st.stderr),
            rep_o.c_code
        );
    }

    // 2. P0-4 & P0-5: 64-bit rev + rbit + clz + udiv/sdiv 除0保护 + uint8_t ~ 取反整型提升 + cc 真编译验证
    let mut in_buf_bytes = Vec::new();
    in_buf_bytes.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
    let asm_u64 = r#"
        ldrb w4, [x0]          // byte 0 = 0x08
        mvn w4, w4             // ~w4 (32-bit)
        ldr x5, [x0]
        rev x5, x5             // BSWAP64
        rbit x6, x5            // RBIT64
        clz x7, x6             // CLZ64
        add x8, x5, x7
        add x8, x8, x4
        mov x9, #0
        udiv x10, x8, x9       // ARM64 div-by-zero -> 0
        add x8, x8, x10
        str x8, [x2]
        ret
    "#;
    let (idx_u, ext_u, _, _, out_u, _) =
        trace_good64_snippet("audit_u64", asm_u64, &in_buf_bytes, &[]);
    let reader_u = TraceReader::open(&idx_u, Some(&ext_u)).unwrap();
    let rep_u = reader_u.decompile_slice(
        &TaintSeed::Mem {
            addr: out_u,
            len: 8,
        },
        &DecompileOptions {
            function_name: "bitops64_fn".to_string(),
            emit_smt2: true,
            ..Default::default()
        },
    );
    assert_eq!(
        rep_u.verified,
        Some(true),
        "64-bit bitops 自校验必须通过:\n{}",
        rep_u.format_summary()
    );
    let smt_u = rep_u.smt2.as_deref().unwrap();
    assert!(smt_u.contains("concat"), "64-bit Bswap/Rbit 应展开为完整 concat");
    assert!(smt_u.contains("ite"), "64-bit Clz 应展开为完整 ite 链");
    if Command::new("cc").arg("--version").output().is_ok() {
        let c_file = idx_u.parent().unwrap().join("bitops64.c");
        let bin_file = idx_u.parent().unwrap().join("bitops64");
        let exp = rep_u.expected_output.unwrap();
        let harness = format!(
            r#"
{c_code}
#include <stdio.h>
int main(void) {{
    uint8_t buf[8] = {{ 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01 }};
    uint64_t r = bitops64_fn(buf);
    if (r != 0x{exp:016x}ULL) {{
        fprintf(stderr, "mismatch: 0x%llx vs 0x{exp:016x}\n", (unsigned long long)r);
        return 1;
    }}
    return 0;
}}
"#,
            c_code = rep_u.c_code
        );
        fs::write(&c_file, harness).unwrap();
        let compile = Command::new("cc")
            .args(["-O2", "-Wall", "-Werror"])
            .arg(&c_file)
            .arg("-o")
            .arg(&bin_file)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "64-bit bitops 生成 C 必须以 -Wall -Werror 编译通过:\n{}\nC:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            rep_u.c_code
        );
        let run = Command::new(&bin_file).output().unwrap();
        assert!(run.status.success());
    }
}
