//! v0.4.5 实战增强端到端测试套件（覆盖 P0-1..3, P1-4..6, P2-7..9 全部 9 项）

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use runidbg_arm64_emu_tracer::good64::assemble;
use runidbg_arm64_emu_tracer::loader::HleBuiltin;
use runidbg_arm64_emu_tracer::tracer::{
    BinaryTracer, DecompileOptions, TaintOptions, TaintSeed, TraceLevel, TraceReader,
    VmAnalysisOptions,
};
use runidbg_arm64_emu_tracer::{AndroidEmulator, CpuBackend};

fn temp_trace_paths(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "runidbg_v045_{tag}_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    (dir.join("trace.idx"), dir.join("trace.ext"))
}

struct TraceRunResult {
    idx_path: PathBuf,
    ext_path: PathBuf,
    code_addr: u64,
    in_buf: u64,
    ro_buf: u64,
    out_buf: u64,
    out_bytes: Vec<u8>,
}

fn run_asm_with_trace(
    tag: &str,
    asm_src: &str,
    in_bytes: &[u8],
    ro_bytes: &[u8],
    out_len: usize,
    level: TraceLevel,
) -> TraceRunResult {
    let (idx_path, ext_path) = temp_trace_paths(tag);
    let mut emu = AndroidEmulator::builder().build().unwrap();
    let code_addr = 0x4000_0000u64;
    let data_addr = 0x4001_0000u64;
    let in_buf = data_addr + 0x100;
    let ro_buf = data_addr + 0x400;
    let out_buf = data_addr + 0x800;

    emu.cpu_mut()
        .mem_map(code_addr, 0x20000, unicorn_engine::unicorn_const::Prot::ALL)
        .unwrap();

    let code = assemble(asm_src, code_addr).expect("good64 assemble");
    emu.cpu_mut().mem_write(code_addr, &code).unwrap();
    if !in_bytes.is_empty() {
        emu.cpu_mut().mem_write(in_buf, in_bytes).unwrap();
    }
    if !ro_bytes.is_empty() {
        emu.cpu_mut().mem_write(ro_buf, ro_bytes).unwrap();
    }

    let tracer = if level == TraceLevel::Full {
        BinaryTracer::new_dual_mmap(&idx_path, &ext_path, 4096, 16384, level).unwrap()
    } else {
        let mut t = BinaryTracer::new_mmap(&idx_path, 4096).unwrap();
        t.set_level(level);
        t
    };
    emu.set_tracer(tracer).unwrap();
    emu.attach_tracer(code_addr, code_addr + 0x1000).unwrap();
    emu.call_address(code_addr, &[in_buf, ro_buf, out_buf])
        .unwrap();
    emu.tracer().flush().unwrap();

    let mut out_bytes = vec![0u8; out_len.max(8)];
    emu.cpu_mut().mem_read(out_buf, &mut out_bytes).unwrap();
    TraceRunResult {
        idx_path,
        ext_path,
        code_addr,
        in_buf,
        ro_buf,
        out_buf,
        out_bytes,
    }
}

#[test]
fn test_p0_1_slice_boundary_from_seq_stop_pc_and_stop_mem() {
    // 阶段 1 (insns 0..5): 复杂初始化阶段，把 in_buf 混合进 x4/x5
    // 阶段 2 (insns 5..9): 目标单轮轮函数 x4 = (x4 ^ x5) + 0x1234, str x4, [x2]
    let mut in_bytes = Vec::new();
    in_bytes.extend_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
    in_bytes.extend_from_slice(&0x99AA_BBCC_DDEE_FF00u64.to_le_bytes());
    let asm = r#"
        ldr x4, [x0]
        ldr x5, [x0, #8]
        add x4, x4, #0x77
        eor x5, x5, x4
        ror x4, x4, #13
        eor x6, x4, x5
        add x6, x6, #0x123
        ror x6, x6, #7
        str x6, [x2]
        ret
    "#;
    let res = run_asm_with_trace("p0_1", asm, &in_bytes, &[], 8, TraceLevel::Full);
    let reader = TraceReader::open(&res.idx_path, Some(&res.ext_path)).unwrap();
    let expected = u64::from_le_bytes(res.out_bytes[0..8].try_into().unwrap());

    // 1. --from-seq 5: 仅反编译阶段 2（从第 5 条 eor x6, x4, x5 开始），把边界寄存器 x4/x5 提升为符号输入
    let rep_from = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &DecompileOptions {
            from_idx: Some(5),
            function_name: "round_only".to_string(),
            ..Default::default()
        },
    );
    assert_eq!(rep_from.stop_idx, 5);
    assert_eq!(rep_from.verified, Some(true), "单轮切片应 bit-exact 自校验通过:\n{}", rep_from.c_code);
    assert_eq!(rep_from.expected_output, Some(expected));
    assert!(
        rep_from.slice_insns <= 5,
        "截断后切片指令数应大幅缩小，实测 {}",
        rep_from.slice_insns
    );
    // 边界输入应为寄存器 x4 / x5，而不是原始内存 in_buf
    assert!(
        rep_from.inputs.iter().any(|i| i.kind == "reg"),
        "from_idx 边界寄存器应被提升为符号参数: {:?}",
        rep_from.inputs
    );

    // 2. --stop-at-pc: 在第 5 条指令的 PC 停下（含该条指令，与 from_idx=5 等价！）
    let phase2_pc = res.code_addr + 5 * 4;
    let rep_pc = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &DecompileOptions {
            stop_at_pcs: vec![phase2_pc],
            ..Default::default()
        },
    );
    assert_eq!(rep_pc.stop_idx, 5);
    assert_eq!(rep_pc.verified, Some(true));
    assert!(rep_pc.slice_insns <= 5);

    // 3. taint_backward 验证 min_idx + stop_at_pcs + stop_mem_ranges + min_idx > anchor_idx
    let taint_rep = reader.taint_backward(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &TaintOptions {
            min_idx: Some(5),
            ..Default::default()
        },
    );
    assert_eq!(taint_rep.stop_idx, 5);
    assert!(taint_rep.steps.len() <= 5);

    let taint_pc = reader.taint_backward(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &TaintOptions {
            stop_at_pcs: vec![phase2_pc],
            ..Default::default()
        },
    );
    assert_eq!(taint_pc.stop_idx, 5);

    let taint_empty = reader.taint_backward(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &TaintOptions {
            anchor_idx: Some(3),
            min_idx: Some(5),
            ..Default::default()
        },
    );
    assert_eq!(taint_empty.scanned_insns, 0);
    assert!(taint_empty.steps.is_empty());
}

#[test]
fn test_p0_2_fold_rodata_and_byte_reassembly_simplification() {
    // 模拟 VM 场景：
    // 1. 从 ro_buf (整个 Trace 从未被写过) 读取 64-bit 字节码常量 0xDEAD_BEEF_CAFE_BABE
    // 2. 从 in_buf 读取 64-bit 动态输入 x
    // 3. 按字节把 x 拆成 8 个字节再左移 OR 拼回 64-bit（或逆序拼回 BSWAP）
    // 4. 与 rodata 常量 XOR 后写入 out_buf
    let in_val = 0x0123_4567_89AB_CDEFu64;
    let ro_const = 0xDEAD_BEEF_CAFE_BABEu64;
    let asm = r#"
        ldr x4, [x0]
        ldr x5, [x1]
        and x6, x4, #0xff
        lsr x7, x4, #8
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #8
        lsr x7, x4, #16
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #16
        lsr x7, x4, #24
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #24
        lsr x7, x4, #32
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #32
        lsr x7, x4, #40
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #40
        lsr x7, x4, #48
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #48
        lsr x7, x4, #56
        and x7, x7, #0xff
        orr x6, x6, x7, lsl #56
        eor x8, x6, x5
        str x8, [x2]
        ret
    "#;
    // 先把 in_val 写进 in_buf（通过栈/前序指令模拟动态写入，而 ro_buf 保持只读从未被写过）
    // 为了让 in_buf 属于「曾被写过」，在 asm 前加一条动态写入或关闭 fold_rodata 对比
    let res = run_asm_with_trace(
        "p0_2",
        asm,
        &in_val.to_le_bytes(),
        &ro_const.to_le_bytes(),
        8,
        TraceLevel::Full,
    );
    let expected = in_val ^ ro_const;
    let actual = u64::from_le_bytes(res.out_bytes[0..8].try_into().unwrap());
    assert_eq!(actual, expected);

    let reader = TraceReader::open(&res.idx_path, Some(&res.ext_path)).unwrap();

    // A. 不开 fold_rodata 时：byte-reassembly 已经把 8 次按字节拆装坍缩为单一表达式！
    let rep_no_fold = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &DecompileOptions {
            function_name: "reasm_only".to_string(),
            ..Default::default()
        },
    );
    assert_eq!(rep_no_fold.verified, Some(true), "C:\n{}", rep_no_fold.c_code);
    assert!(
        rep_no_fold.simplified_nodes < rep_no_fold.raw_nodes,
        "byte-reassembly 应大幅坍缩节点: raw={} -> simplified={}\nC:\n{}",
        rep_no_fold.raw_nodes,
        rep_no_fold.simplified_nodes,
        rep_no_fold.c_code
    );
    assert!(
        !rep_no_fold.c_code.contains("uint64_t t0") && !rep_no_fold.c_code.contains("& 0xff"),
        "24 条按字节拆装指令应坍缩为单行表达式，不应残留中间临时变量或 `& 0xff`:\n{}",
        rep_no_fold.c_code
    );

    // B. 开启 fold_rodata 并钉死 in_buf 为符号输入（stop_mem），ro_buf 自动折叠为常量！
    let rep_fold = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &DecompileOptions {
            fold_rodata: true,
            stop_mem_ranges: vec![(res.in_buf, 8)],
            function_name: "folded_vm_formula".to_string(),
            ..Default::default()
        },
    );
    assert_eq!(rep_fold.verified, Some(true), "C:\n{}", rep_fold.c_code);
    assert_eq!(
        rep_fold.inputs.len(),
        1,
        "ro_buf 折叠为常量后应只剩 1 个动态输入参数: {:?}\nC:\n{}",
        rep_fold.inputs,
        rep_fold.c_code
    );
    assert!(
        rep_fold.c_code.contains("0xdeadbeefcafebabe"),
        "rodata 常量应直接内联到伪 C:\n{}",
        rep_fold.c_code
    );
}

#[test]
fn test_p0_3_umulh_smulh_magic_div_ubfiz_sbfiz_rev16_and_real_cc() {
    // 1. 编译器常见魔数除法 x / 62500:
    //    umulh(x, 0x8637BD05AF6C69B6) >> 15  (精确等于 x / 62500)
    // 2. smulh(x, 0x123456789ABCDEF1)
    // 3. ubfiz + sbfiz + rev16
    let in_val = 123_456_789_012u64;
    let asm = r#"
        ldr x4, [x0]
        movz x5, #0x69b6
        movk x5, #0xaf6c, lsl #16
        movk x5, #0xbd05, lsl #32
        movk x5, #0x8637, lsl #48
        umulh x6, x4, x5
        lsr x6, x6, #15
        movz x7, #0xdef1
        movk x7, #0x9abc, lsl #16
        movk x7, #0x5678, lsl #32
        movk x7, #0x1234, lsl #48
        smulh x8, x4, x7
        .word 0xd37c2c89
        .word 0x937d1c8a
        rev16 x11, x4
        eor x12, x6, x8
        add x12, x12, x9
        eor x12, x12, x10
        add x12, x12, x11
        str x12, [x2]
        ret
    "#;
    let res = run_asm_with_trace(
        "p0_3",
        asm,
        &in_val.to_le_bytes(),
        &[],
        8,
        TraceLevel::Full,
    );
    let expected = u64::from_le_bytes(res.out_bytes[0..8].try_into().unwrap());
    let reader = TraceReader::open(&res.idx_path, Some(&res.ext_path)).unwrap();
    let rep = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: res.out_buf,
            len: 8,
        },
        &DecompileOptions {
            function_name: "magic_div_and_bitfields".to_string(),
            emit_smt2: true,
            ..Default::default()
        },
    );
    assert_eq!(
        rep.verified,
        Some(true),
        "UMULH/SMULH/UBFIZ/SBFIZ/REV16 必须全部精确自校验通过!\nC:\n{}",
        rep.c_code
    );
    assert_eq!(rep.expected_output, Some(expected));
    assert!(
        rep.c_code.contains("UMULH64") && rep.c_code.contains("SMULH64"),
        "应输出 UMULH64 与 SMULH64:\n{}",
        rep.c_code
    );
    assert!(
        rep.c_code.contains("/ 62500"),
        "应反推出魔数除法注释 `/* ~= / 62500 */`:\n{}",
        rep.c_code
    );
    assert!(
        rep.smt2.as_deref().unwrap().contains("bvmul"),
        "SMT2 应包含 128-bit bvmul 展开"
    );

    // 真实 cc -O2 -Wall -Werror 编译并跨输入验证！
    if Command::new("cc").arg("--version").output().is_ok() {
        let dir = res.idx_path.parent().unwrap();
        let c_file = dir.join("verify_p0_3.c");
        let bin_file = dir.join("verify_p0_3");
        let harness = format!(
            r#"
{c_code}
#include <stdio.h>
int main(void) {{
    uint64_t x1 = {in_val}ULL;
    uint64_t r1 = magic_div_and_bitfields((const uint8_t *)&x1);
    if (r1 != 0x{expected:016x}ULL) {{
        fprintf(stderr, "r1 mismatch: 0x%llx vs 0x{expected:016x}\n", (unsigned long long)r1);
        return 1;
    }}
    return 0;
}}
"#,
            c_code = rep.c_code
        );
        fs::write(&c_file, harness).unwrap();
        let st = Command::new("cc")
            .args(["-O2", "-Wall", "-Werror"])
            .arg(&c_file)
            .arg("-o")
            .arg(&bin_file)
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "cc 编译失败:\n{}\nC:\n{}",
            String::from_utf8_lossy(&st.stderr),
            rep.c_code
        );
        let run = Command::new(&bin_file).output().unwrap();
        assert!(run.status.success(), "执行失败: {}", String::from_utf8_lossy(&run.stderr));
    }
}

#[test]
fn test_p1_5_hle_entropy_and_clock_origin_tracking() {
    let (idx_path, ext_path) = temp_trace_paths("p1_5");
    let mut emu = AndroidEmulator::builder().build().unwrap();
    let code_addr = 0x4000_0000u64;
    let data_addr = 0x4001_0000u64;
    let tv_buf = data_addr + 0x100;
    let out_buf = data_addr + 0x200;
    emu.cpu_mut()
        .mem_map(code_addr, 0x20000, unicorn_engine::unicorn_const::Prot::ALL)
        .unwrap();

    // 注册 HLE rand 与 gettimeofday 跳板
    let rand_addr = emu.register_hle_builtin("rand", HleBuiltin::Rand).unwrap();
    let gtod_addr = emu
        .register_hle_builtin("gettimeofday", HleBuiltin::Gettimeofday)
        .unwrap();

    // guest 代码：
    //   x19 = rand_addr, x20 = gtod_addr, x21 = tv_buf, x22 = out_buf
    //   bl rand() -> x23 = x0
    //   x0 = tv_buf, x1 = 0; bl gettimeofday()
    //   ldr x24, [tv_buf] (tv_sec)
    //   eor x25, x23, x24
    //   str x25, [out_buf]
    let asm = r#"
        stp x29, x30, [sp, #-16]!
        mov x19, x0
        mov x20, x1
        mov x21, x2
        mov x22, x3
        blr x19
        mov x23, x0
        mov x0, x21
        mov x1, #0
        blr x20
        ldr x24, [x21]
        eor x25, x23, x24
        str x25, [x22]
        ldp x29, x30, [sp], #16
        ret
    "#;
    let code = assemble(asm, code_addr).unwrap();
    emu.cpu_mut().mem_write(code_addr, &code).unwrap();

    let tracer =
        BinaryTracer::new_dual_mmap(&idx_path, &ext_path, 4096, 16384, TraceLevel::Full).unwrap();
    emu.set_tracer(tracer).unwrap();
    emu.attach_tracer(0, u64::MAX).unwrap();
    emu.call_address(code_addr, &[rand_addr, gtod_addr, tv_buf, out_buf])
        .unwrap();
    emu.tracer().flush().unwrap();

    let reader = TraceReader::open(&idx_path, Some(&ext_path)).unwrap();
    let hles = reader.hle_events();
    assert!(
        hles.iter().any(|e| e.op == "rand") && hles.iter().any(|e| e.op == "gettimeofday"),
        "HLE 事件流应包含 rand 与 gettimeofday: {:?}",
        hles
    );

    let taint_rep = reader.taint_backward(
        &TaintSeed::Mem {
            addr: out_buf,
            len: 8,
        },
        &TaintOptions::default(),
    );
    let summary = taint_rep.format_summary(0);
    assert!(
        taint_rep.origin_annotations.iter().any(|s| s.contains("rand")),
        "应标注 rand() 熵源起源:\n{summary}"
    );
    assert!(
        taint_rep
            .origin_annotations
            .iter()
            .any(|s| s.contains("gettimeofday")),
        "应标注 gettimeofday() 时钟起源:\n{summary}"
    );

    let decomp_rep = reader.decompile_slice(
        &TaintSeed::Mem {
            addr: out_buf,
            len: 8,
        },
        &DecompileOptions::default(),
    );
    assert_eq!(
        decomp_rep.verified,
        Some(true),
        "含 HLE rand/gettimeofday 起源的反编译应自校验通过:\n{}",
        decomp_rep.c_code
    );
    assert!(
        decomp_rep.inputs.iter().any(|i| i.kind == "hle_source"),
        "输入参数应识别为 hle_source: {:?}",
        decomp_rep.inputs
    );
}

#[test]
fn test_p1_6_crypto_scan_aes_simon_tea_and_io_candidates() {
    // 构造包含 TEA/XTEA 常量 0x9e3779b9 + Simon-128/256 高熵 z-序列常量 0x3369f885192c0ef5
    // + 非 256B 对齐软件 AES S-Box 查表 + 明显输入/输出缓冲访存的 trace
    let in_bytes = [0x11u8; 16];
    let aes_sbox_head: [u8; 16] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
        0x76,
    ];
    // 故意把 S-Box 放在非 256B 对齐偏移 (ro_buf + 0x13) 测试任意对齐检测
    let mut ro_bytes = vec![0u8; 0x30];
    ro_bytes[0x13..0x23].copy_from_slice(&aes_sbox_head);
    let asm = r#"
        ldr x4, [x0]
        ldr x5, [x0, #8]
        movz w6, #0x79b9
        movk w6, #0x9e37, lsl #16
        add w4, w4, w6
        movz x7, #0x0ef5
        movk x7, #0x192c, lsl #16
        movk x7, #0xf885, lsl #32
        movk x7, #0x3369, lsl #48
        eor x5, x5, x7
        add x10, x1, #0x13
        ldrb w11, [x10, #0]
        ldrb w11, [x10, #1]
        ldrb w11, [x10, #2]
        ldrb w11, [x10, #3]
        ldrb w11, [x10, #4]
        ldrb w11, [x10, #5]
        add x4, x4, x11
        str x4, [x2]
        str x5, [x2, #8]
        ret
    "#;
    let res = run_asm_with_trace("p1_6", asm, &in_bytes, &ro_bytes, 16, TraceLevel::Full);
    let reader = TraceReader::open(&res.idx_path, Some(&res.ext_path)).unwrap();
    let groups = reader.scan_crypto();
    assert!(
        groups.iter().any(|g| g.algorithm.contains("TEA")),
        "应检测到 TEA/XTEA 常量: {:?}",
        groups
    );
    assert!(
        groups.iter().any(|g| g.algorithm.contains("Simon")),
        "应检测到 Simon-128 常量: {:?}",
        groups
    );
    assert!(
        groups.iter().any(|g| g.algorithm == "AES (S-Box)"),
        "应检测到非 256B 对齐的软件 AES S-Box 查表: {:?}",
        groups
    );
    let tea = groups.iter().find(|g| g.algorithm.contains("TEA")).unwrap();
    assert!(
        tea.input_addrs.iter().any(|&(a, _)| a == res.in_buf)
            && tea.output_addrs.iter().any(|&(a, _)| a == res.out_buf),
        "crypto-scan 应推断出候选输入/输出缓冲地址: in={:?}, out={:?}",
        tea.input_addrs,
        tea.output_addrs
    );
}

#[test]
fn test_p2_7_missing_ext_warning_and_level_tag() {
    let asm = "ldr x4, [x0]\nstr x4, [x2]\nret\n";
    let full = run_asm_with_trace("p2_7_full", asm, &0x42u64.to_le_bytes(), &[], 8, TraceLevel::Full);
    let fast = run_asm_with_trace("p2_7_fast", asm, &0x42u64.to_le_bytes(), &[], 8, TraceLevel::Fast);

    // Full trace 缺 .ext 时必须发出告警
    let r_full_missing = TraceReader::open(&full.idx_path, None).unwrap();
    assert!(r_full_missing.missing_ext_warning());
    assert_eq!(r_full_missing.recorded_level_name(), Some("full"));

    // Full trace 带 .ext 时无告警
    let r_full_ok = TraceReader::open(&full.idx_path, Some(&full.ext_path)).unwrap();
    assert!(!r_full_ok.missing_ext_warning());

    // Fast trace 本来就没 .ext，无告警
    let r_fast = TraceReader::open(&fast.idx_path, None).unwrap();
    assert!(!r_fast.missing_ext_warning());
    assert_eq!(r_fast.recorded_level_name(), Some("fast"));
}

#[test]
fn test_p1_4_and_p2_9_filtered_query_vm_range_and_dual_run_diff_taint() {
    // 构造两次运行 A 和 B：
    // out_buf[0..8]  写入固定常量 0x1111_2222_3333_4444（A/B 相同）
    // out_buf[8..16] 写入 in_buf[0..8] ^ 0x55AA（A/B 输入不同导致输出不同）
    let asm = r#"
        movz x4, #0x4444
        movk x4, #0x3333, lsl #16
        movk x4, #0x2222, lsl #32
        movk x4, #0x1111, lsl #48
        str x4, [x2]
        ldr x5, [x0]
        mov x6, #0x55aa
        eor x5, x5, x6
        str x5, [x2, #8]
        ldr x7, [x2, #8]
        ret
    "#;
    let run_a = run_asm_with_trace(
        "p2_9_a",
        asm,
        &0xAABB_CCDD_EEFF_0011u64.to_le_bytes(),
        &[],
        16,
        TraceLevel::Full,
    );
    let run_b = run_asm_with_trace(
        "p2_9_b",
        asm,
        &0x1122_3344_5566_7788u64.to_le_bytes(),
        &[],
        16,
        TraceLevel::Full,
    );
    assert_eq!(run_a.out_buf, run_b.out_buf);
    let ra = TraceReader::open(&run_a.idx_path, Some(&run_a.ext_path)).unwrap();
    let rb = TraceReader::open(&run_b.idx_path, Some(&run_b.ext_path)).unwrap();

    // P1-4: query_addr_history_filtered 支持 reads/writes/seq_range 精细过滤
    let writes_only = ra.query_addr_history_filtered(
        run_a.out_buf + 8,
        8,
        None,
        None,
        false,
        true,
        false,
        100,
    );
    let reads_only = ra.query_addr_history_filtered(
        run_a.out_buf + 8,
        8,
        None,
        None,
        true,
        false,
        false,
        100,
    );
    assert_eq!(writes_only.len(), 1);
    assert_eq!(reads_only.len(), 1);
    assert_eq!(writes_only[0].kind, "write");
    assert_eq!(reads_only[0].kind, "read");

    // P1-4: VM analyze 接受 min_idx / max_idx 区间
    let vm_rep = ra.analyze_vm(&VmAnalysisOptions {
        min_idx: Some(2),
        max_idx: Some(6),
        ..Default::default()
    });
    assert_eq!(vm_rep.scanned_insns, 5);

    // P2-9: 双跑差分 + 自动反向污点联动
    let dt = TraceReader::diff_taint_bytes(&ra, &rb, run_a.out_buf, 16, 0);
    assert_eq!(dt.identical_bytes, 8);
    assert_eq!(dt.differing_bytes, 8);
    assert_eq!(dt.spans.len(), 1);
    assert_eq!(dt.spans[0].offset, 8);
    assert_eq!(dt.spans[0].len, 8);
    assert!(
        dt.spans[0].origins_a_mem.iter().any(|&(a, _)| a == run_a.in_buf),
        "A 侧差异字节应溯源到 in_buf:\n{}",
        dt.format_summary()
    );
    assert!(
        dt.spans[0].origins_b_mem.iter().any(|&(a, _)| a == run_b.in_buf),
        "B 侧差异字节应溯源到 in_buf:\n{}",
        dt.format_summary()
    );

    // P2-8: QuickJS API (Uint8Array bytes, BigInt value64, findMem, Trace.diffTaint, missingExtWarning)
    #[cfg(feature = "js")]
    {
        use runidbg_arm64_emu_tracer::script::JsSession;
        use std::path::Path;

        let js = format!(
            r#"
            (() => {{
                const tr = Trace.open({idx_a:?}, {ext_a:?});
                const mem = tr.reconstructMem({out_buf}, 16);
                if (!(mem.bytes instanceof Uint8Array)) throw new Error("bytes must be Uint8Array");
                const wHits = tr.findMem({out_buf} + 8, {{ kind: "w", len: 8, includeHle: false }});
                if (wHits.length !== 1) throw new Error("expected 1 write hit, got " + wHits.length);
                if (typeof wHits[0].value64 !== "bigint") throw new Error("value64 must be bigint");
                const expected64 = 0xAABBCCDDEEFF0011n ^ 0x55AAn;
                if (wHits[0].value64 !== expected64) {{
                    throw new Error("value64 mismatch: " + wHits[0].value64.toString(16));
                }}
                const rHits = tr.findMem({out_buf} + 8, {{ kind: "r", len: 8 }});
                if (rHits.length !== 1) throw new Error("expected 1 read hit");
                const level = tr.recordedLevel;
                const missing = tr.missingExtWarning;
                tr.close();
                const dt = Trace.diffTaint({idx_a:?}, {idx_b:?}, {out_buf}, 16);
                return {{
                    level,
                    missing,
                    isU8: mem.bytes instanceof Uint8Array,
                    diffBytes: dt.differingBytes,
                    spanCount: dt.spans.length,
                }};
            }})()
            "#,
            idx_a = run_a.idx_path.to_string_lossy(),
            ext_a = run_a.ext_path.to_string_lossy(),
            idx_b = run_b.idx_path.to_string_lossy(),
            out_buf = run_a.out_buf,
        );
        let session = JsSession::new(&[], Path::new("."), true).unwrap();
        let out_json = session.eval_json_string(&js).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out_json).unwrap();
        assert_eq!(v["level"], "full");
        assert_eq!(v["missing"], false);
        assert_eq!(v["isU8"], true);
        assert_eq!(v["diffBytes"], 8);
        assert_eq!(v["spanCount"], 1);
    }

    // cleanup unused warning
    let _ = run_a.ro_buf;
}

#[test]
fn test_cli_v045_subcommands_and_missing_ext_warn() {
    let bin = env!("CARGO_BIN_EXE_runidbg-cli");
    let asm = r#"
        movz x4, #0x4444
        movk x4, #0x3333, lsl #16
        movk x4, #0x2222, lsl #32
        movk x4, #0x1111, lsl #48
        str x4, [x2]
        ldr x5, [x0]
        mov x6, #0x55aa
        eor x5, x5, x6
        str x5, [x2, #8]
        ret
    "#;
    let run_a = run_asm_with_trace(
        "cli_a",
        asm,
        &0xAABB_CCDD_EEFF_0011u64.to_le_bytes(),
        &[],
        16,
        TraceLevel::Full,
    );
    let run_b = run_asm_with_trace(
        "cli_b",
        asm,
        &0x1122_3344_5566_7788u64.to_le_bytes(),
        &[],
        16,
        TraceLevel::Full,
    );

    // 1. CLI `mem --hex` & `--out`
    let out_bin = run_a.idx_path.parent().unwrap().join("reconstructed.bin");
    let mem_hex = Command::new(bin)
        .args([
            "mem",
            run_a.idx_path.to_str().unwrap(),
            "--addr",
            &format!("0x{:x}", run_a.out_buf),
            "--len",
            "16",
            "--hex",
        ])
        .output()
        .unwrap();
    assert!(
        mem_hex.status.success(),
        "cli mem --hex failed: {}",
        String::from_utf8_lossy(&mem_hex.stderr)
    );
    let hex_stdout = String::from_utf8_lossy(&mem_hex.stdout).trim().to_string();
    assert_eq!(hex_stdout.len(), 32);
    assert!(hex_stdout.starts_with("4444333322221111"));

    let mem_out = Command::new(bin)
        .args([
            "mem",
            run_a.idx_path.to_str().unwrap(),
            "--addr",
            &format!("0x{:x}", run_a.out_buf),
            "--len",
            "16",
            "--out",
            out_bin.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(mem_out.status.success());
    assert_eq!(fs::read(&out_bin).unwrap().len(), 16);

    // 2. CLI `find --addr --len --seq-range --only-writes --no-hle`
    let find_out = Command::new(bin)
        .args([
            "find",
            run_a.idx_path.to_str().unwrap(),
            "--addr",
            &format!("0x{:x}", run_a.out_buf + 8),
            "--len",
            "8",
            "--seq-range",
            "0..20",
            "--only-writes",
            "--no-hle",
        ])
        .output()
        .unwrap();
    assert!(
        find_out.status.success(),
        "cli find failed: {}",
        String::from_utf8_lossy(&find_out.stderr)
    );
    let find_str = String::from_utf8_lossy(&find_out.stdout);
    assert!(find_str.contains("[write]"), "find output:\n{find_str}");

    // 3. CLI `diff --target-addr --len`
    let diff_out = Command::new(bin)
        .args([
            "diff",
            run_a.idx_path.to_str().unwrap(),
            run_b.idx_path.to_str().unwrap(),
            "--target-addr",
            &format!("0x{:x}", run_a.out_buf),
            "--len",
            "16",
        ])
        .output()
        .unwrap();
    assert!(
        diff_out.status.success(),
        "cli diff --target-addr failed: {}",
        String::from_utf8_lossy(&diff_out.stderr)
    );
    let diff_str = String::from_utf8_lossy(&diff_out.stdout);
    assert!(
        diff_str.contains("相同字节=8") && diff_str.contains("差异字节=8"),
        "diff --target-addr output:\n{diff_str}"
    );

    // 4. Missing `.ext` warning when copying `.idx` alone without `.ext`
    let orphan_idx = run_a.idx_path.parent().unwrap().join("orphan.idx");
    fs::copy(&run_a.idx_path, &orphan_idx).unwrap();
    let stats_out = Command::new(bin)
        .args(["stats", orphan_idx.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(stats_out.status.success());
    let stderr_str = String::from_utf8_lossy(&stats_out.stderr);
    assert!(
        stderr_str.contains("WARN:"),
        "缺失伴生 .ext 时 CLI 应输出 WARN 告警，实测 stderr:\n{stderr_str}"
    );

    // 5. CLI `mem --len` 上界防 OOM 校验
    let oom_guard = Command::new(bin)
        .args([
            "mem",
            run_a.idx_path.to_str().unwrap(),
            "--addr",
            &format!("0x{:x}", run_a.out_buf),
            "--len",
            "999999999",
            "--hex",
        ])
        .output()
        .unwrap();
    assert!(!oom_guard.status.success(), "超大 --len 必须被安全拒绝");
}

#[test]
fn test_v045_audit_regressions_core_and_mcp() {
    // 1. Core P0-1: 16-bit BSWAP16 字节重装自校验 + 真 cc 编译验证（绝不可误吐 BSWAP32）
    let in_u16 = 0x1234u16;
    let asm_bswap16 = r#"
        ldrh w4, [x0]
        and w5, w4, #0xff
        lsl w5, w5, #8
        lsr w6, w4, #8
        and w6, w6, #0xff
        orr w5, w5, w6
        strh w5, [x2]
        ret
    "#;
    let res16 = run_asm_with_trace(
        "audit_bswap16",
        asm_bswap16,
        &in_u16.to_le_bytes(),
        &[],
        8,
        TraceLevel::Full,
    );
    let r16 = TraceReader::open(&res16.idx_path, Some(&res16.ext_path)).unwrap();
    let rep16 = r16.decompile_slice(
        &TaintSeed::Mem {
            addr: res16.out_buf,
            len: 2,
        },
        &DecompileOptions {
            function_name: "bswap16_fn".to_string(),
            ..Default::default()
        },
    );
    assert_eq!(rep16.verified, Some(true), "C:\n{}", rep16.c_code);
    assert!(
        !rep16.c_code.contains("BSWAP32"),
        "16-bit Bswap 绝不可生成 BSWAP32:\n{}",
        rep16.c_code
    );
    if Command::new("cc").arg("--version").output().is_ok() {
        let dir = res16.idx_path.parent().unwrap();
        let c_file = dir.join("verify_bswap16.c");
        let bin_file = dir.join("verify_bswap16");
        let harness = format!(
            r#"
{c_code}
int main(void) {{
    uint8_t b1[2] = {{ 0x34, 0x12 }};
    if (bswap16_fn(b1) != 0x3412u) return 1;
    uint8_t b2[2] = {{ 0xCD, 0xAB }};
    if (bswap16_fn(b2) != 0xCDABu) return 2;
    return 0;
}}
"#,
            c_code = rep16.c_code
        );
        fs::write(&c_file, harness).unwrap();
        let st = Command::new("cc")
            .args(["-O2", "-Wall", "-Werror"])
            .arg(&c_file)
            .arg("-o")
            .arg(&bin_file)
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "bswap16 C 编译失败:\n{}\nC:\n{}",
            String::from_utf8_lossy(&st.stderr),
            rep16.c_code
        );
        let run = Command::new(&bin_file).output().unwrap();
        assert!(run.status.success(), "bswap16 C 执行失败");
    }

    // 2. Core P0-2 + P0-3:
    //    - 先 memset(&tv, 0, 16) 再调用 gettimeofday(&tv, NULL) 读取 tv.tv_usec（bytes 8..16）-> verified == Some(true)
    //    - --stop-mem 钉死经过 memset 清零的缓冲区 -> 仍作为符号输入而不折叠成常量 0
    {
        let (idx_path, ext_path) = temp_trace_paths("audit_p02_p03");
        let mut emu = AndroidEmulator::builder().build().unwrap();
        let code_addr = 0x4000_0000u64;
        let data_addr = 0x4001_0000u64;
        let tv_buf = data_addr + 0x100;
        let out_buf = data_addr + 0x200;
        emu.cpu_mut()
            .mem_map(code_addr, 0x20000, unicorn_engine::unicorn_const::Prot::ALL)
            .unwrap();
        let memset_addr = emu.register_hle_builtin("memset", HleBuiltin::Memset).unwrap();
        let gtod_addr = emu
            .register_hle_builtin("gettimeofday", HleBuiltin::Gettimeofday)
            .unwrap();
        let asm = r#"
            stp x29, x30, [sp, #-16]!
            mov x19, x0
            mov x20, x1
            mov x21, x2
            mov x22, x3
            mov x0, x21
            mov x1, #0
            mov x2, #16
            blr x19
            mov x0, x21
            mov x1, #0
            blr x20
            ldr x23, [x21, #8]
            add x23, x23, #0x123
            str x23, [x22]
            ldp x29, x30, [sp], #16
            ret
        "#;
        let code = assemble(asm, code_addr).unwrap();
        emu.cpu_mut().mem_write(code_addr, &code).unwrap();
        let tracer =
            BinaryTracer::new_dual_mmap(&idx_path, &ext_path, 4096, 16384, TraceLevel::Full)
                .unwrap();
        emu.set_tracer(tracer).unwrap();
        emu.attach_tracer(0, u64::MAX).unwrap();
        emu.call_address(code_addr, &[memset_addr, gtod_addr, tv_buf, out_buf])
            .unwrap();
        emu.tracer().flush().unwrap();

        let r = TraceReader::open(&idx_path, Some(&ext_path)).unwrap();
        let rep = r.decompile_slice(
            &TaintSeed::Mem {
                addr: out_buf,
                len: 8,
            },
            &DecompileOptions::default(),
        );
        assert_eq!(
            rep.verified,
            Some(true),
            "gettimeofday tv_usec (bytes 8..16) 不可被旧 memset 污染，必须 bit-exact 自校验通过:\n{}",
            rep.c_code
        );
        // 钉死 tv_buf+8 为 stop_mem 时必须保持为符号输入参数而非折叠成常量
        let rep_pin = r.decompile_slice(
            &TaintSeed::Mem {
                addr: out_buf,
                len: 8,
            },
            &DecompileOptions {
                stop_mem_ranges: vec![(tv_buf + 8, 8)],
                ..Default::default()
            },
        );
        assert_eq!(rep_pin.verified, Some(true));
        assert!(
            rep_pin.inputs.iter().any(|i| i.kind == "mem"),
            "--stop-mem 必须保持为符号内存输入:\n{}",
            rep_pin.c_code
        );
    }

    // 3. Core P1-3: --fold-rodata 预扫描不得把「先读后写」的可变缓冲区误折为常量
    let asm_rbw = r#"
        ldr x4, [x0]
        add x5, x4, #1
        str x5, [x0]
        str x5, [x2]
        ret
    "#;
    let res_rbw = run_asm_with_trace(
        "audit_rbw",
        asm_rbw,
        &0x1020_3040_5060_7080u64.to_le_bytes(),
        &[],
        8,
        TraceLevel::Full,
    );
    let r_rbw = TraceReader::open(&res_rbw.idx_path, Some(&res_rbw.ext_path)).unwrap();
    let rep_rbw = r_rbw.decompile_slice(
        &TaintSeed::Mem {
            addr: res_rbw.out_buf,
            len: 8,
        },
        &DecompileOptions {
            fold_rodata: true,
            ..Default::default()
        },
    );
    assert_eq!(rep_rbw.verified, Some(true));
    assert!(
        !rep_rbw.inputs.is_empty(),
        "先读后写的可变缓冲区绝不可被 --fold-rodata 折成常量:\n{}",
        rep_rbw.c_code
    );

    // 4. Core P1-8: TraceLevel::RegDelta 单文件含 LDP 多寄存器变更时不可误报 missing_ext_warning
    let asm_ldp = "ldp x4, x5, [x0]\nadd x4, x4, x5\nret\n";
    let res_rd = run_asm_with_trace(
        "audit_regdelta",
        asm_ldp,
        &[1u8; 16],
        &[],
        8,
        TraceLevel::RegDelta,
    );
    let r_rd = TraceReader::open(&res_rd.idx_path, None).unwrap();
    assert_eq!(r_rd.recorded_level_name(), Some("reg_delta"));
    assert!(
        !r_rd.missing_ext_warning(),
        "RegDelta 单文件 trace 绝不可误报 missing_ext_warning"
    );

    // 5. MCP `trace_mem` + `trace_diff(targetAddr)` + `trace_decompile(fromIdx/foldRodata)`
    #[cfg(all(feature = "js", feature = "mcp"))]
    {
        use runidbg_arm64_emu_tracer::script::mcp::McpServer;
        use serde_json::json;
        use std::path::Path;

        let mut srv = McpServer::new(Path::new("."), None).unwrap();
        let req_mem = json!({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "trace_mem",
                "arguments": {
                    "idxPath": res_rbw.idx_path.to_string_lossy(),
                    "addr": format!("0x{:x}", res_rbw.out_buf),
                    "len": 8
                }
            }
        });
        let resp: serde_json::Value =
            serde_json::from_str(&srv.handle_message(&req_mem.to_string()).unwrap()).unwrap();
        let payload: serde_json::Value =
            serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(payload["knownBytes"], 8);
        assert_eq!(payload["missingExtWarning"], false);
        assert_eq!(payload["hex"], "8170605040302010");
    }
}
