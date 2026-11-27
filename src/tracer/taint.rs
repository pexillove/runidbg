use std::collections::{BTreeMap, BTreeSet, HashMap};

use bad64::{Operand, Reg};

use crate::tracer::reader::TraceReader;
use crate::tracer::record::{ExtKind, ExtRecord, HleTraceEvent};

/// 污点分析追踪的总寄存器槽位数：0..=30 为 X0..X30，31 为 SP，32..=63 为 V0..V31 (NEON/FP)
const NUM_REG_SLOTS: usize = 64;

/// 污点种子 / 反向切片目标
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaintSeed {
    /// 内存区间 `[addr, addr + len)`
    Mem { addr: u64, len: usize },
    /// 寄存器槽位（0..=30 = X0..X30, 31 = SP, 32..=63 = V0..V31）
    Reg { reg_id: u8 },
}

impl TaintSeed {
    pub fn describe(&self) -> String {
        match *self {
            Self::Mem { addr, len } => format!("mem[0x{addr:016x}+{len}]"),
            Self::Reg { reg_id } => format_reg_slot(reg_id),
        }
    }
}

/// 离线污点传播 / 反向切片配置
#[derive(Debug, Clone)]
pub struct TaintOptions {
    /// Forward: 起始指令下标（默认 0）；Backward: 目标观测点下标（默认最后一条指令）
    pub anchor_idx: Option<usize>,
    /// Backward 模式下可选切片起点下界（含该指令；等价 `--from-seq`），精准截断到单轮循环或子阶段
    pub min_idx: Option<usize>,
    /// Backward 模式下命中这些 PC 时立即停止回溯（把当前边界寄存器/内存提升为输入起源）
    pub stop_at_pcs: Vec<u64>,
    /// Backward 模式下遇到读取这些内存区间 `(addr, len)` 时直接钉死为输入起源，不再追寻更早写入
    pub stop_mem_ranges: Vec<(u64, usize)>,
    /// 最大扫描指令数（0 表示不限，扫描至 Trace 边界）
    pub max_insns: usize,
    /// 报告中最多保留的传播步骤条数（默认 512）
    pub max_steps: usize,
    /// 是否追踪访存变址/动态地址寄存器（如 S-Box 查表 `LDRB Wd, [Xtable, Xidx]` 或 `ADD+LDR`，默认 true）
    pub track_mem_index: bool,
    /// 是否追踪栈指针 `SP` 本身（默认 false，避免 SP 算术污染整个栈帧；若种子本身是 SP 则自动开启）
    pub track_sp: bool,
}

impl Default for TaintOptions {
    fn default() -> Self {
        Self {
            anchor_idx: None,
            min_idx: None,
            stop_at_pcs: Vec::new(),
            stop_mem_ranges: Vec::new(),
            max_insns: 0,
            max_steps: 512,
            track_mem_index: true,
            track_sp: false,
        }
    }
}

/// 单条污点传播 / 反向切片步骤（每条均绑定 `(idx, seq, pc)`，完全免疫栈帧与 VM 寄存器堆地址复用）
#[derive(Debug, Clone)]
pub struct TaintStep {
    pub idx: usize,
    pub seq: u32,
    pub pc: u64,
    pub asm: String,
    pub kind: &'static str,
    pub sources: Vec<String>,
    pub dests: Vec<String>,
    /// Forward 模式下为种子字节偏移位掩码（bit i 表示 seed 第 i%64 字节）；Backward 模式下为 0
    pub tag_mask: u64,
    pub value: Option<u64>,
}

/// 离线污点传播 / 反向切片报告
#[derive(Debug, Clone)]
pub struct TaintReport {
    /// `"forward"` 或 `"backward"`
    pub mode: &'static str,
    pub seed_desc: String,
    pub anchor_idx: usize,
    /// 实际停止边界指令下标（Backward 模式为回溯停下的下界指令下标；Forward 模式为结束下标）
    pub stop_idx: usize,
    pub scanned_insns: usize,
    pub total_matching_steps: usize,
    /// 按时间顺序排列的因果传播步骤（Backward 模式已反转为「起源 -> 目标」顺序）
    pub steps: Vec<TaintStep>,
    /// Forward: 结束时仍带污点的寄存器 `(name, tag_mask)`；Backward: 溯源起点未解出的输入寄存器
    pub tainted_regs: Vec<(String, u64)>,
    /// Forward: 结束时仍带污点的内存区间 `(start, len, tag_mask)`；Backward: 溯源起点未解出的输入内存区间
    pub tainted_mem_ranges: Vec<(u64, usize, u64)>,
    /// Backward 模式下识别到的外部熵源/时钟源 HLE 起源标注（如 `[ORIGIN: HLE rand() #seq -> 0x...]`）
    pub origin_annotations: Vec<String>,
}

impl TaintReport {
    /// 生成面向终端 / 日志的可读污点传播图摘要
    pub fn format_summary(&self, base_addr: u64) -> String {
        let mut out = String::new();
        let fmt_rel = |pc: u64| -> String {
            if base_addr != 0 && pc >= base_addr {
                format!(" [+0x{:x}]", pc - base_addr)
            } else {
                String::new()
            }
        };
        let trunc_note = if self.total_matching_steps > self.steps.len() {
            if self.mode == "backward" {
                format!("，已截断：仅展示靠近目标的最后 {} 步", self.steps.len())
            } else {
                format!("，已截断：仅展示靠近种子的前 {} 步", self.steps.len())
            }
        } else {
            String::new()
        };
        out.push_str(&format!(
            "# 离线污点分析 ({}) | 种子: {} | 锚点: #{} | 扫描指令: {} | 命中步数: {} (展示 {}{trunc_note})\n",
            self.mode,
            self.seed_desc,
            self.anchor_idx,
            self.scanned_insns,
            self.total_matching_steps,
            self.steps.len()
        ));
        for s in &self.steps {
            let val_str = s.value.map(|v| format!(" =0x{v:x}")).unwrap_or_default();
            let mask_str = if s.tag_mask != 0 {
                format!(" tag=0x{:x}", s.tag_mask)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "  #{:06} pc=0x{:016x}{} [{:<9}] {:<36} | [{}] -> [{}]{val_str}{mask_str}\n",
                s.idx,
                s.pc,
                fmt_rel(s.pc),
                s.kind,
                s.asm,
                s.sources.join(", "),
                s.dests.join(", "),
            ));
        }
        let end_label = if self.mode == "forward" {
            "最终受污染状态"
        } else {
            "溯源输入起源"
        };
        if !self.tainted_regs.is_empty() {
            let regs: Vec<String> = self
                .tainted_regs
                .iter()
                .map(|(n, m)| {
                    if *m != 0 {
                        format!("{n}(0x{m:x})")
                    } else {
                        n.clone()
                    }
                })
                .collect();
            out.push_str(&format!("# {end_label} - 寄存器: {}\n", regs.join(", ")));
        }
        if !self.tainted_mem_ranges.is_empty() {
            let mems: Vec<String> = self
                .tainted_mem_ranges
                .iter()
                .map(|(a, l, m)| {
                    if *m != 0 {
                        format!("0x{a:016x}+{l}(0x{m:x})")
                    } else {
                        format!("0x{a:016x}+{l}")
                    }
                })
                .collect();
            out.push_str(&format!("# {end_label} - 内存区间: {}\n", mems.join(", ")));
        }
        if !self.origin_annotations.is_empty() {
            out.push_str(&format!(
                "# 外部熵源/时钟源起源标注:\n{}\n",
                self.origin_annotations
                    .iter()
                    .map(|s| format!("  {s}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        out
    }
}

/// 将寄存器名称（如 `x0`/`w8`/`sp`/`fp`/`lr`/`v0`/`q3`/`d1`）解析为内部槽位 ID (`0..=63`)
pub fn parse_reg_slot(name: &str) -> Option<u8> {
    let lower = name.trim().to_ascii_lowercase();
    match lower.as_str() {
        "sp" | "wsp" => Some(31),
        "fp" => Some(29),
        "lr" => Some(30),
        "xzr" | "wzr" => None,
        _ => {
            let mut chars = lower.chars();
            let prefix = chars.next()?;
            let num: u8 = chars.as_str().parse().ok()?;
            match prefix {
                'x' | 'w' if num <= 30 => Some(num),
                'v' | 'q' | 'd' | 's' | 'h' | 'b' if num <= 31 => Some(32 + num),
                _ => None,
            }
        }
    }
}

/// 将内部槽位 ID (`0..=63`) 格式化为规范 ARM64 寄存器名
pub fn format_reg_slot(slot: u8) -> String {
    match slot {
        0..=30 => format!("X{slot}"),
        31 => "SP".to_string(),
        32..=63 => format!("V{}", slot - 32),
        other => format!("R{other}"),
    }
}

#[inline]
fn reg_to_slot(reg: Reg, track_sp: bool) -> Option<u8> {
    let slot = parse_reg_slot(reg.name())?;
    if slot == 31 && !track_sp {
        None
    } else {
        Some(slot)
    }
}

#[inline]
fn operand_has_lane(op: &Operand) -> bool {
    match op {
        Operand::Reg {
            arrspec: Some(a), ..
        }
        | Operand::MultiReg {
            arrspec: Some(a), ..
        } => a.lane().is_some(),
        _ => false,
    }
}

#[inline]
fn operand_reg_slot(op: &Operand, track_sp: bool) -> Option<u8> {
    match op {
        Operand::Reg { reg, .. }
        | Operand::ShiftReg { reg, .. }
        | Operand::QualReg { reg, .. } => reg_to_slot(*reg, track_sp),
        _ => None,
    }
}

#[inline]
fn operand_elem_size(op: &Operand) -> usize {
    match op {
        Operand::Reg { reg, .. } => reg.size().max(4),
        _ => 8,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomicKind {
    /// `CAS Ws, Wt, [Xn]` / `CASP Xs, Xs1, Xt, Xt1, [Xn]`：op0(s) 读旧内存值，op1(t) 写入内存
    Cas { is_pair: bool },
    /// `SWP Ws, Wt, [Xn]`：op0 覆写内存（不混入旧内存污点），op1 读旧内存值
    Swp,
    /// `LDADD*`/`LDCLR*`/`LDEOR*`/`LDSET*` 等：op0 与旧内存混合写回内存，op1 读旧内存值
    LdOp,
    /// `STADD*`/`STCLR*`/`STEOR*`/`STSET*` 等：op0 与旧内存混合写回内存，无目的寄存器
    StOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsnCategory {
    /// 不传播寄存器/内存数据污点（纯分支/比较/屏障）
    NoDataFlow,
    /// `BL`/`BLR`：写 `LR(X30)` 为常量返回地址（并结合 HLE 事件处理 `X0`）
    Call,
    /// 纯常量生成（`MOVZ`/`MOVN`/`ADR`/`ADRP`/`CSET`/`CSETM`）：清除目标寄存器污点
    ConstDef,
    /// 加载类指令（含 `LDR*`/`LDP`/`LDPSW`/`LDNP`/`LDUR*`/`LDAR*`/`LDAPR*`/`LDXR*`/`LD1..4`）
    Load { is_pair: bool, is_rmw_lane: bool },
    /// 存储类指令（含 `STR*`/`STP`/`STNP`/`STUR*`/`STLR*`/`STLUR*`/`STXR*`/`ST1..4`）
    Store { is_pair: bool, has_status_reg: bool },
    /// 原子读改写指令（细分 `Cas` / `Swp` / `LdOp` / `StOp`）
    AtomicRmw(AtomicKind),
    /// 目标寄存器同时作为源操作数（`MOVK`/`BFI`/`BFC`/`MOV vN.B[i]`/`AESE`/`SHA256*`/`SM3*`/`SM4*` 等）
    AluRmw,
    /// 常规 ALU / SIMD 数据处理：op[0] 为写目标，op[1..] 为源操作数
    AluOverwrite,
}

struct DecodedSemantics {
    category: InsnCategory,
    /// 目的寄存器槽位（Load/Alu/ConstDef/AtomicRmw/Store status）
    dst_regs: Vec<u8>,
    /// 源值寄存器槽位（Store 值寄存器 / ALU 源寄存器 / AtomicRmw 源寄存器）
    src_regs: Vec<u8>,
    /// 对加载/存储（`LDP`/`LDPSW`/`STP`/`STNP`/`LDXP`/`STXP`）保留定长 2 槽（`XZR`/`WZR` 为 `None`）
    pair_regs: [Option<u8>; 2],
    /// 对加载/存储的单元素字节宽度（4 / 8 / 16）
    pair_elem_size: usize,
    /// Forward 模式下的访存动态地址寄存器（含 `MemExt` 双寄存器及 `MemReg`/`MemOffset` 基址）
    fwd_mem_addr_regs: Vec<u8>,
    /// Backward 模式下的显式变址寄存器（`MemExt` 索引寄存器，如 S-Box `[Xtable, Xidx]` 中的 `Xidx`）
    bwd_mem_index_regs: Vec<u8>,
    /// `MemPostIdxReg([base, stride])` 的基址写回依赖 `(base_slot, stride_slot)`
    post_idx_writeback: Option<(u8, u8)>,
}

fn collect_operand_regs(op: &Operand, track_sp: bool, out: &mut Vec<u8>) {
    match op {
        Operand::Reg { reg, .. }
        | Operand::ShiftReg { reg, .. }
        | Operand::QualReg { reg, .. } => {
            if let Some(s) = reg_to_slot(*reg, track_sp) {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        Operand::MultiReg { regs, .. } => {
            for r in regs.iter().flatten() {
                if let Some(s) = reg_to_slot(*r, track_sp) {
                    if !out.contains(&s) {
                        out.push(s);
                    }
                }
            }
        }
        _ => {}
    }
}

fn extract_mem_addr_deps(
    op: &Operand,
    track_sp: bool,
    fwd_addr: &mut Vec<u8>,
    bwd_idx: &mut Vec<u8>,
    post_wb: &mut Option<(u8, u8)>,
) {
    match op {
        Operand::MemExt { regs, .. } => {
            // Forward 检查 regs[0] 与 regs[1]（兼容编译器交换 `[Xidx, Xtable]`，无污点表基址零误报）
            for &r in regs {
                if let Some(s) = reg_to_slot(r, track_sp) {
                    if !fwd_addr.contains(&s) {
                        fwd_addr.push(s);
                    }
                }
            }
            // Backward 追踪规范索引寄存器 regs[1]
            if let Some(s) = reg_to_slot(regs[1], track_sp) {
                if !bwd_idx.contains(&s) {
                    bwd_idx.push(s);
                }
            }
        }
        Operand::MemReg(reg)
        | Operand::MemOffset { reg, .. }
        | Operand::MemPreIdx { reg, .. }
        | Operand::MemPostIdxImm { reg, .. } => {
            // Forward 允许检测被污点污染的基址指针（如 `ADD X7, Xtable, Xidx; LDRB W6, [X7]`）
            if let Some(s) = reg_to_slot(*reg, track_sp) {
                if !fwd_addr.contains(&s) {
                    fwd_addr.push(s);
                }
            }
        }
        Operand::MemPostIdxReg(regs) => {
            // `LD1 {v0}, [x0], x1`: x0 是本次访存地址，x1 仅用于加载后的 `x0 += x1` 写回，不是本次访存索引！
            if let Some(base) = reg_to_slot(regs[0], track_sp) {
                if !fwd_addr.contains(&base) {
                    fwd_addr.push(base);
                }
                if let Some(stride) = reg_to_slot(regs[1], track_sp) {
                    *post_wb = Some((base, stride));
                }
            }
        }
        _ => {}
    }
}

fn classify_mnemonic(mnem: &str) -> InsnCategory {
    // 1. 纯控制流、比较、屏障、系统提示
    if matches!(
        mnem,
        "b" | "br"
            | "braa"
            | "brab"
            | "braaz"
            | "brabz"
            | "ret"
            | "retaa"
            | "retab"
            | "cbz"
            | "cbnz"
            | "tbz"
            | "tbnz"
            | "cmp"
            | "cmn"
            | "tst"
            | "ccmp"
            | "ccmn"
            | "fcmp"
            | "fcmpe"
            | "fccmp"
            | "fccmpe"
            | "nop"
            | "dmb"
            | "dsb"
            | "isb"
            | "prfm"
            | "prfum"
            | "hint"
            | "yield"
            | "wfe"
            | "wfi"
            | "sev"
            | "sevl"
            | "svc"
            | "brk"
            | "hlt"
            | "clrex"
            | "bti"
            | "autiasp"
            | "paciasp"
            | "autibsp"
            | "pacibsp"
    ) || mnem.starts_with("b.")
    {
        return InsnCategory::NoDataFlow;
    }

    if matches!(mnem, "bl" | "blr" | "blraa" | "blrab" | "blraaz" | "blrabz") {
        return InsnCategory::Call;
    }

    // 2. 纯常量定义
    if matches!(mnem, "movz" | "movn" | "adr" | "adrp" | "cset" | "csetm" | "mrs") {
        return InsnCategory::ConstDef;
    }

    // 3. 原子读改写（细分子类，准确区分操作数方向）
    if mnem.starts_with("casp") {
        return InsnCategory::AtomicRmw(AtomicKind::Cas { is_pair: true });
    }
    if mnem.starts_with("cas") {
        return InsnCategory::AtomicRmw(AtomicKind::Cas { is_pair: false });
    }
    if mnem.starts_with("swp") {
        return InsnCategory::AtomicRmw(AtomicKind::Swp);
    }
    if mnem.starts_with("ldadd")
        || mnem.starts_with("ldclr")
        || mnem.starts_with("ldeor")
        || mnem.starts_with("ldset")
        || mnem.starts_with("ldsmax")
        || mnem.starts_with("ldsmin")
        || mnem.starts_with("ldumax")
        || mnem.starts_with("ldumin")
    {
        return InsnCategory::AtomicRmw(AtomicKind::LdOp);
    }
    if mnem.starts_with("stadd")
        || mnem.starts_with("stclr")
        || mnem.starts_with("steor")
        || mnem.starts_with("stset")
        || mnem.starts_with("stsmax")
        || mnem.starts_with("stsmin")
        || mnem.starts_with("stumax")
        || mnem.starts_with("stumin")
    {
        return InsnCategory::AtomicRmw(AtomicKind::StOp);
    }

    // 4. 加载指令
    if matches!(mnem, "ldp" | "ldpsw" | "ldnp" | "ldxp" | "ldaxp") {
        return InsnCategory::Load {
            is_pair: true,
            is_rmw_lane: false,
        };
    }
    if mnem.starts_with("ldr")
        || mnem.starts_with("ldur")
        || mnem.starts_with("ldar")
        || mnem.starts_with("ldapr")
        || mnem.starts_with("ldapur")
        || mnem.starts_with("ldlar")
        || mnem.starts_with("ldxr")
        || mnem.starts_with("ldaxr")
        || mnem.starts_with("ldtr")
        || mnem.starts_with("ld1")
        || mnem.starts_with("ld2")
        || mnem.starts_with("ld3")
        || mnem.starts_with("ld4")
    {
        return InsnCategory::Load {
            is_pair: false,
            is_rmw_lane: false,
        };
    }

    // 5. 存储指令
    if matches!(mnem, "stp" | "stnp") {
        return InsnCategory::Store {
            is_pair: true,
            has_status_reg: false,
        };
    }
    if matches!(mnem, "stxp" | "stlxp") {
        return InsnCategory::Store {
            is_pair: true,
            has_status_reg: true,
        };
    }
    if mnem.starts_with("stxr") || mnem.starts_with("stlxr") {
        return InsnCategory::Store {
            is_pair: false,
            has_status_reg: true,
        };
    }
    if mnem.starts_with("str")
        || mnem.starts_with("stur")
        || mnem.starts_with("stlr")
        || mnem.starts_with("stlur")
        || mnem.starts_with("stllr")
        || mnem.starts_with("sttr")
        || mnem.starts_with("st1")
        || mnem.starts_with("st2")
        || mnem.starts_with("st3")
        || mnem.starts_with("st4")
    {
        return InsnCategory::Store {
            is_pair: false,
            has_status_reg: false,
        };
    }

    // 6. 读-改-写目标寄存器的 ALU / Crypto / SIMD / PAC 指令
    //    注意排除完全覆写目的寄存器的 `sha1h` / `sm3ss1` / `sm4ekey`
    let is_crypto_rmw = (mnem.starts_with("sha1") && mnem != "sha1h")
        || mnem.starts_with("sha256")
        || mnem.starts_with("sha512")
        || (mnem.starts_with("sm3") && mnem != "sm3ss1")
        || (mnem.starts_with("sm4") && mnem != "sm4ekey");
    let is_pac_rmw = mnem.starts_with("pac") || mnem.starts_with("aut") || mnem.starts_with("xpac");

    if matches!(
        mnem,
        "movk"
            | "bfi"
            | "bfc"
            | "bfxil"
            | "bfm"
            | "ins"
            | "tbx"
            | "sli"
            | "sri"
            | "aese"
            | "aesd"
            | "bit"
            | "bif"
            | "bsl"
            | "mla"
            | "mls"
            | "fmla"
            | "fmls"
            | "fmlal"
            | "fmlal2"
            | "fmlsl"
            | "fmlsl2"
            | "saba"
            | "uaba"
            | "sabal"
            | "sabal2"
            | "uabal"
            | "uabal2"
            | "sadalp"
            | "uadalp"
            | "smlal"
            | "smlal2"
            | "umlal"
            | "umlal2"
            | "smlsl"
            | "smlsl2"
            | "umlsl"
            | "umlsl2"
            | "sdot"
            | "udot"
            | "usdot"
            | "sudot"
            | "sqrdmlah"
            | "sqrdmlsh"
    ) || mnem.starts_with("sqdmlal")
        || mnem.starts_with("sqdmlsl")
        || is_crypto_rmw
        || is_pac_rmw
    {
        return InsnCategory::AluRmw;
    }

    InsnCategory::AluOverwrite
}

fn decode_semantics(opcode: u32, pc: u64, track_sp: bool) -> (String, DecodedSemantics) {
    let Ok(insn) = bad64::decode(opcode, pc) else {
        return (
            format!(".word 0x{opcode:08x}"),
            DecodedSemantics {
                category: InsnCategory::NoDataFlow,
                dst_regs: Vec::new(),
                src_regs: Vec::new(),
                pair_regs: [None, None],
                pair_elem_size: 8,
                fwd_mem_addr_regs: Vec::new(),
                bwd_mem_index_regs: Vec::new(),
                post_idx_writeback: None,
            },
        );
    };
    let asm = insn.to_string();
    let mnem = insn.op().mnem();
    let ops = insn.operands();
    let mut category = classify_mnemonic(mnem);

    // bad64 将 NEON `INS vN.<T>[lane], ...` 反汇编为 `mov vN.<T>[lane], ...`，必须归为 AluRmw 防止误杀其他 lane
    if mnem == "mov" && ops.first().is_some_and(operand_has_lane) {
        category = InsnCategory::AluRmw;
    }
    // `LD1 {vN.<T>}[lane], [Xn]` 仅改写单个 lane，保留其他字节
    if let InsnCategory::Load { is_pair: false, .. } = category {
        if ops.first().is_some_and(operand_has_lane) {
            category = InsnCategory::Load {
                is_pair: false,
                is_rmw_lane: true,
            };
        }
    }

    let mut dst_regs = Vec::new();
    let mut src_regs = Vec::new();
    let mut pair_regs = [None, None];
    let mut pair_elem_size = 8usize;
    let mut fwd_mem_addr_regs = Vec::new();
    let mut bwd_mem_index_regs = Vec::new();
    let mut post_idx_writeback = None;

    match category {
        InsnCategory::NoDataFlow => {}
        InsnCategory::Call => {
            dst_regs.push(30);
        }
        InsnCategory::ConstDef => {
            if let Some(op0) = ops.first() {
                collect_operand_regs(op0, track_sp, &mut dst_regs);
            }
        }
        InsnCategory::Load { is_pair, .. } => {
            if is_pair {
                if let Some(op0) = ops.first() {
                    pair_regs[0] = operand_reg_slot(op0, track_sp);
                    pair_elem_size = if mnem == "ldpsw" {
                        4
                    } else {
                        operand_elem_size(op0)
                    };
                    collect_operand_regs(op0, track_sp, &mut dst_regs);
                }
                if let Some(op1) = ops.get(1) {
                    pair_regs[1] = operand_reg_slot(op1, track_sp);
                    collect_operand_regs(op1, track_sp, &mut dst_regs);
                }
                if let Some(mem_op) = ops.get(2) {
                    extract_mem_addr_deps(
                        mem_op,
                        track_sp,
                        &mut fwd_mem_addr_regs,
                        &mut bwd_mem_index_regs,
                        &mut post_idx_writeback,
                    );
                }
            } else {
                if let Some(op0) = ops.first() {
                    collect_operand_regs(op0, track_sp, &mut dst_regs);
                }
                if let Some(mem_op) = ops.get(1) {
                    extract_mem_addr_deps(
                        mem_op,
                        track_sp,
                        &mut fwd_mem_addr_regs,
                        &mut bwd_mem_index_regs,
                        &mut post_idx_writeback,
                    );
                }
            }
        }
        InsnCategory::Store {
            is_pair,
            has_status_reg,
        } => {
            let val_start = usize::from(has_status_reg);
            if has_status_reg {
                if let Some(op0) = ops.first() {
                    collect_operand_regs(op0, track_sp, &mut dst_regs);
                }
            }
            if is_pair {
                if let Some(op_a) = ops.get(val_start) {
                    pair_regs[0] = operand_reg_slot(op_a, track_sp);
                    pair_elem_size = operand_elem_size(op_a);
                    collect_operand_regs(op_a, track_sp, &mut src_regs);
                }
                if let Some(op_b) = ops.get(val_start + 1) {
                    pair_regs[1] = operand_reg_slot(op_b, track_sp);
                    collect_operand_regs(op_b, track_sp, &mut src_regs);
                }
                if let Some(mem_op) = ops.get(val_start + 2) {
                    extract_mem_addr_deps(
                        mem_op,
                        track_sp,
                        &mut fwd_mem_addr_regs,
                        &mut bwd_mem_index_regs,
                        &mut post_idx_writeback,
                    );
                }
            } else {
                if let Some(op_v) = ops.get(val_start) {
                    collect_operand_regs(op_v, track_sp, &mut src_regs);
                }
                if let Some(mem_op) = ops.get(val_start + 1) {
                    extract_mem_addr_deps(
                        mem_op,
                        track_sp,
                        &mut fwd_mem_addr_regs,
                        &mut bwd_mem_index_regs,
                        &mut post_idx_writeback,
                    );
                }
            }
        }
        InsnCategory::AtomicRmw(kind) => match kind {
            AtomicKind::Cas { is_pair: false } => {
                // CAS Ws, Wt, [Xn]: Ws(op0) 接收旧内存值, Wt(op1) 写入内存
                if let Some(op0) = ops.first() {
                    collect_operand_regs(op0, track_sp, &mut dst_regs);
                }
                if let Some(op1) = ops.get(1) {
                    collect_operand_regs(op1, track_sp, &mut src_regs);
                }
            }
            AtomicKind::Cas { is_pair: true } => {
                // CASP Xs, Xs1, Xt, Xt1, [Xn]
                for op in ops.iter().take(2) {
                    collect_operand_regs(op, track_sp, &mut dst_regs);
                }
                for op in ops.iter().skip(2).take(2) {
                    collect_operand_regs(op, track_sp, &mut src_regs);
                }
            }
            AtomicKind::Swp | AtomicKind::LdOp => {
                // SWP/LDADD Xs, Xt, [Xn]: Xs(op0) 为源, Xt(op1) 接收旧内存值
                if let Some(op0) = ops.first() {
                    collect_operand_regs(op0, track_sp, &mut src_regs);
                }
                if let Some(op1) = ops.get(1) {
                    collect_operand_regs(op1, track_sp, &mut dst_regs);
                }
            }
            AtomicKind::StOp => {
                // STADD/STCLR/STEOR/STSET Xs, [Xn]: Xs(op0) 为源，无目的寄存器
                if let Some(op0) = ops.first() {
                    collect_operand_regs(op0, track_sp, &mut src_regs);
                }
            }
        },
        InsnCategory::AluRmw => {
            if let Some(op0) = ops.first() {
                collect_operand_regs(op0, track_sp, &mut dst_regs);
                collect_operand_regs(op0, track_sp, &mut src_regs);
            }
            for op in ops.iter().skip(1) {
                collect_operand_regs(op, track_sp, &mut src_regs);
            }
        }
        InsnCategory::AluOverwrite => {
            // 自清零惯用法识别（同时支持标量 GPR 与 NEON 向量 `eor v0.16b, v0.16b, v0.16b`）
            let is_self_zero = matches!(mnem, "eor" | "sub" | "subs" | "bic" | "bics")
                && ops.len() == 3
                && matches!(
                    (&ops[1], &ops[2]),
                    (
                        Operand::Reg { reg: r1, arrspec: a1 },
                        Operand::Reg { reg: r2, arrspec: a2 }
                    ) if r1 == r2 && a1 == a2
                );
            if let Some(op0) = ops.first() {
                collect_operand_regs(op0, track_sp, &mut dst_regs);
            }
            if is_self_zero {
                category = InsnCategory::ConstDef;
            } else {
                for op in ops.iter().skip(1) {
                    collect_operand_regs(op, track_sp, &mut src_regs);
                }
            }
        }
    }

    (
        asm,
        DecodedSemantics {
            category,
            dst_regs,
            src_regs,
            pair_regs,
            pair_elem_size,
            fwd_mem_addr_regs,
            bwd_mem_index_regs,
            post_idx_writeback,
        },
    )
}

/// 将 `LDP`/`STP` 产生的 1..4 条拆包访存事件按地址精确划分到槽 0 (`[base, base+elem)`) 与槽 1 (`[base+elem, base+2*elem)`)
fn partition_pair_mem_events(
    events: &[ExtRecord],
    elem_size: usize,
) -> [Vec<(u64, usize, u64)>; 2] {
    let mut slots: [Vec<(u64, usize, u64)>; 2] = [Vec::new(), Vec::new()];
    if events.is_empty() || elem_size == 0 {
        return slots;
    }
    let base = events.iter().map(|e| e.addr).min().unwrap_or(0);
    let split = base.saturating_add(elem_size as u64);
    let total_end = split.saturating_add(elem_size as u64);
    for &e in events {
        let addr = e.addr;
        let sz = e.size_or_reg.max(1) as usize;
        let val = e.value;
        let ev_end = addr.saturating_add(sz as u64);
        // 槽 0 交集: [addr, ev_end) ∩ [base, split)
        let lo0 = addr.max(base);
        let hi0 = ev_end.min(split);
        if lo0 < hi0 {
            slots[0].push((lo0, (hi0 - lo0) as usize, val));
        }
        // 槽 1 交集: [addr, ev_end) ∩ [split, total_end)
        let lo1 = addr.max(split);
        let hi1 = ev_end.min(total_end);
        if lo1 < hi1 {
            let shift_bytes = (lo1 - addr) as usize;
            let shifted_val = if shift_bytes < 8 {
                val >> (shift_bytes * 8)
            } else {
                0
            };
            slots[1].push((lo1, (hi1 - lo1) as usize, shifted_val));
        }
    }
    slots
}

fn coalesce_mem_tags(mem_taint: &HashMap<u64, u64>) -> Vec<(u64, usize, u64)> {
    let sorted: BTreeMap<u64, u64> = mem_taint
        .iter()
        .filter(|(_, &tag)| tag != 0)
        .map(|(&a, &t)| (a, t))
        .collect();
    let mut out: Vec<(u64, usize, u64)> = Vec::new();
    for (addr, tag) in sorted {
        if let Some(last) = out.last_mut() {
            if last.0.saturating_add(last.1 as u64) == addr && last.2 == tag {
                last.1 += 1;
                continue;
            }
        }
        out.push((addr, 1, tag));
    }
    out
}

fn coalesce_mem_set(addrs: &BTreeSet<u64>) -> Vec<(u64, usize, u64)> {
    let mut out: Vec<(u64, usize, u64)> = Vec::new();
    for &addr in addrs {
        if let Some(last) = out.last_mut() {
            if last.0.saturating_add(last.1 as u64) == addr {
                last.1 += 1;
                continue;
            }
        }
        out.push((addr, 1, 0));
    }
    out
}

fn mem_range_tag(mem_taint: &HashMap<u64, u64>, addr: u64, size: usize) -> u64 {
    let mut mask = 0u64;
    for off in 0..size {
        if let Some(a) = addr.checked_add(off as u64) {
            if let Some(&t) = mem_taint.get(&a) {
                mask |= t;
            }
        }
    }
    mask
}

fn set_mem_range_tag(mem_taint: &mut HashMap<u64, u64>, addr: u64, size: usize, tag: u64) {
    if tag == 0 {
        for off in 0..size {
            if let Some(a) = addr.checked_add(off as u64) {
                mem_taint.remove(&a);
            }
        }
    } else {
        for off in 0..size {
            if let Some(a) = addr.checked_add(off as u64) {
                mem_taint.insert(a, tag);
            }
        }
    }
}

fn split_mem_events(events: &[ExtRecord]) -> (Vec<ExtRecord>, Vec<ExtRecord>) {
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for &e in events {
        if e.kind == ExtKind::MemRead as u8 {
            reads.push(e);
        } else if e.kind == ExtKind::MemWrite as u8 {
            writes.push(e);
        }
    }
    (reads, writes)
}

fn lookup_reg_delta(reader: &TraceReader, idx: usize, reg_slot: u8) -> Option<u64> {
    if reg_slot > 31 {
        return None;
    }
    reader
        .reg_deltas_for(idx)
        .into_iter()
        .find(|&(rid, _)| rid == reg_slot)
        .map(|(_, val)| val)
}

/// 判断该 HLE 事件是否将全新返回值（指针/状态码/熵源/时间戳）写入 X0（不含 `memcpy`/`memmove` 透传原 `dst`）
#[inline]
pub(crate) fn hle_defines_fresh_x0(op: &str) -> bool {
    matches!(
        op,
        "malloc"
            | "calloc"
            | "realloc"
            | "posix_memalign"
            | "free"
            | "bzero"
            | "rand"
            | "random"
            | "lrand48"
            | "arc4random"
            | "time"
    )
}

#[inline]
pub(crate) fn is_hle_entropy_or_time_reg(op: &str) -> bool {
    matches!(op, "rand" | "random" | "lrand48" | "arc4random" | "time")
}

#[inline]
pub(crate) fn is_hle_entropy_or_time_mem(op: &str) -> bool {
    matches!(op, "getrandom" | "gettimeofday" | "clock_gettime")
}

impl TraceReader {
    /// 离线前向污点追踪（Forward Taint Analysis）
    ///
    /// 从 `seed`（内存字节区间或寄存器）在 `opts.anchor_idx`（默认 0）处染色出发，
    /// 逐条沿 ARM64 指令语义、访存流及 HLE `memcpy/memmove/memset/realloc` 传播污点，
    /// 当同一地址被无污点数据覆写时自动清除旧污点，彻底避免栈帧/VM 寄存器堆复用导致的误报。
    pub fn taint_forward(&self, seed: &TaintSeed, opts: &TaintOptions) -> TaintReport {
        let total = self.len();
        let start_idx = opts.anchor_idx.unwrap_or(0).min(total);
        let end_idx = if opts.max_insns > 0 {
            start_idx.saturating_add(opts.max_insns).min(total)
        } else {
            total
        };
        let max_steps = if opts.max_steps == 0 {
            usize::MAX
        } else {
            opts.max_steps
        };
        // 若种子本身是 SP (31)，自动开启 track_sp 防止静默空跑
        let track_sp = opts.track_sp || matches!(seed, TaintSeed::Reg { reg_id: 31 });

        let mut reg_taint = [0u64; NUM_REG_SLOTS];
        let mut mem_taint: HashMap<u64, u64> = HashMap::new();

        match *seed {
            TaintSeed::Mem { addr, len } => {
                for off in 0..len {
                    if let Some(a) = addr.checked_add(off as u64) {
                        let bit = 1u64 << (off % 64);
                        mem_taint.insert(a, bit);
                    }
                }
            }
            TaintSeed::Reg { reg_id } => {
                if (reg_id as usize) < NUM_REG_SLOTS {
                    reg_taint[reg_id as usize] = 1;
                }
            }
        }

        let mut steps: Vec<TaintStep> = Vec::new();
        let mut total_matching_steps = 0usize;

        let mut push_step = |step: TaintStep| {
            total_matching_steps += 1;
            if steps.len() < max_steps {
                steps.push(step);
            }
        };

        for idx in start_idx..end_idx {
            let Some(rec) = self.get(idx) else {
                break;
            };
            let pc = rec.pc;
            let opcode = rec.opcode;
            let seq = rec.seq_low32;

            let (asm, sem) = decode_semantics(opcode, pc, track_sp);
            let mem_evts = self.mem_events_for(idx);
            let (mem_reads, mem_writes) = split_mem_events(&mem_evts);
            let hle_evts = self.hle_events_for(idx);

            match sem.category {
                InsnCategory::NoDataFlow => {}
                InsnCategory::Call => {
                    // BL/BLR 写入返回地址到 LR(X30)，清除 LR 污点
                    reg_taint[30] = 0;
                }
                InsnCategory::ConstDef => {
                    for &d in &sem.dst_regs {
                        reg_taint[d as usize] = 0;
                    }
                }
                InsnCategory::Load {
                    is_pair,
                    is_rmw_lane,
                } => {
                    let addr_tag = if opts.track_mem_index {
                        sem.fwd_mem_addr_regs
                            .iter()
                            .fold(0u64, |acc, &r| acc | reg_taint[r as usize])
                    } else {
                        0
                    };

                    if is_pair {
                        let slot_reads = partition_pair_mem_events(&mem_reads, sem.pair_elem_size);
                        let mut sources = Vec::new();
                        let mut dests = Vec::new();
                        let mut combined_tag = 0u64;
                        let mut first_dst = None;
                        for slot_i in 0..2 {
                            let Some(dst) = sem.pair_regs[slot_i] else {
                                continue;
                            };
                            let mut mem_tag = 0u64;
                            for &(a, sz, _) in &slot_reads[slot_i] {
                                let t = mem_range_tag(&mem_taint, a, sz);
                                if t != 0 {
                                    mem_tag |= t;
                                    sources.push(format!("mem[0x{a:016x}+{sz}]"));
                                }
                            }
                            let tag = mem_tag | addr_tag;
                            reg_taint[dst as usize] = tag;
                            if tag != 0 {
                                combined_tag |= tag;
                                dests.push(format_reg_slot(dst));
                                if first_dst.is_none() {
                                    first_dst = Some(dst);
                                }
                            }
                        }
                        if combined_tag != 0 {
                            if addr_tag != 0 {
                                for &r in &sem.fwd_mem_addr_regs {
                                    if reg_taint[r as usize] != 0 {
                                        sources.push(format_reg_slot(r));
                                    }
                                }
                            }
                            let val = first_dst.and_then(|d| lookup_reg_delta(self, idx, d));
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "load",
                                sources,
                                dests,
                                tag_mask: combined_tag,
                                value: val,
                            });
                        }
                    } else {
                        let mut mem_tag = 0u64;
                        let mut sources = Vec::new();
                        for r in &mem_reads {
                            let sz = r.size_or_reg.max(1) as usize;
                            let r_addr = r.addr;
                            let t = mem_range_tag(&mem_taint, r_addr, sz);
                            if t != 0 {
                                mem_tag |= t;
                                sources.push(format!("mem[0x{r_addr:016x}+{sz}]"));
                            }
                        }
                        if addr_tag != 0 {
                            for &r in &sem.fwd_mem_addr_regs {
                                if reg_taint[r as usize] != 0 {
                                    sources.push(format_reg_slot(r));
                                }
                            }
                        }
                        let incoming_tag = mem_tag | addr_tag;
                        let mut dests = Vec::new();
                        let mut out_tag = 0u64;
                        for &d in &sem.dst_regs {
                            let new_tag = if is_rmw_lane {
                                reg_taint[d as usize] | incoming_tag
                            } else {
                                incoming_tag
                            };
                            reg_taint[d as usize] = new_tag;
                            if incoming_tag != 0 {
                                out_tag |= new_tag;
                                dests.push(format_reg_slot(d));
                            }
                        }
                        if incoming_tag != 0 && !dests.is_empty() {
                            let val = sem
                                .dst_regs
                                .first()
                                .and_then(|&d| lookup_reg_delta(self, idx, d));
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "load",
                                sources,
                                dests,
                                tag_mask: out_tag,
                                value: val,
                            });
                        }
                    }

                    // 处理 `MemPostIdxReg([base, stride])` 写回：`base = base + stride`
                    if let Some((base, stride)) = sem.post_idx_writeback {
                        let s_tag = reg_taint[stride as usize];
                        if s_tag != 0 {
                            reg_taint[base as usize] |= s_tag;
                        }
                    }
                }
                InsnCategory::Store {
                    is_pair,
                    has_status_reg,
                } => {
                    if has_status_reg {
                        for &d in &sem.dst_regs {
                            reg_taint[d as usize] = 0;
                        }
                    }
                    if is_pair {
                        let slot_writes =
                            partition_pair_mem_events(&mem_writes, sem.pair_elem_size);
                        let mut sources = Vec::new();
                        let mut dests = Vec::new();
                        let mut combined_tag = 0u64;
                        let mut first_val = None;
                        for slot_i in 0..2 {
                            // XZR/WZR 槽位为 None，tag = 0 会清除对应目标内存字节污点
                            let tag = sem.pair_regs[slot_i]
                                .map(|r| reg_taint[r as usize])
                                .unwrap_or(0);
                            for &(a, sz, v) in &slot_writes[slot_i] {
                                set_mem_range_tag(&mut mem_taint, a, sz, tag);
                                if tag != 0 {
                                    combined_tag |= tag;
                                    if let Some(sr) = sem.pair_regs[slot_i] {
                                        let sname = format_reg_slot(sr);
                                        if !sources.contains(&sname) {
                                            sources.push(sname);
                                        }
                                    }
                                    dests.push(format!("mem[0x{a:016x}+{sz}]"));
                                    if first_val.is_none() {
                                        first_val = Some(v);
                                    }
                                }
                            }
                        }
                        if combined_tag != 0 {
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "store",
                                sources,
                                dests,
                                tag_mask: combined_tag,
                                value: first_val,
                            });
                        }
                    } else {
                        let tag = sem
                            .src_regs
                            .iter()
                            .fold(0u64, |acc, &r| acc | reg_taint[r as usize]);
                        let mut dests = Vec::new();
                        let mut first_val = None;
                        for w in &mem_writes {
                            let sz = w.size_or_reg.max(1) as usize;
                            let w_addr = w.addr;
                            let w_val = w.value;
                            set_mem_range_tag(&mut mem_taint, w_addr, sz, tag);
                            if tag != 0 {
                                dests.push(format!("mem[0x{w_addr:016x}+{sz}]"));
                                if first_val.is_none() {
                                    first_val = Some(w_val);
                                }
                            }
                        }
                        if tag != 0 && !dests.is_empty() {
                            let sources: Vec<String> = sem
                                .src_regs
                                .iter()
                                .filter(|&&r| reg_taint[r as usize] != 0)
                                .map(|&r| format_reg_slot(r))
                                .collect();
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "store",
                                sources,
                                dests,
                                tag_mask: tag,
                                value: first_val,
                            });
                        }
                    }

                    if let Some((base, stride)) = sem.post_idx_writeback {
                        let s_tag = reg_taint[stride as usize];
                        if s_tag != 0 {
                            reg_taint[base as usize] |= s_tag;
                        }
                    }
                }
                InsnCategory::AtomicRmw(akind) => {
                    let mut mem_tag = 0u64;
                    let mut sources = Vec::new();
                    for r in &mem_reads {
                        let sz = r.size_or_reg.max(1) as usize;
                        let r_addr = r.addr;
                        let t = mem_range_tag(&mem_taint, r_addr, sz);
                        if t != 0 {
                            mem_tag |= t;
                            sources.push(format!("mem[0x{r_addr:016x}+{sz}]"));
                        }
                    }
                    let src_tag = sem
                        .src_regs
                        .iter()
                        .fold(0u64, |acc, &r| acc | reg_taint[r as usize]);
                    for &r in &sem.src_regs {
                        if reg_taint[r as usize] != 0 {
                            sources.push(format_reg_slot(r));
                        }
                    }
                    // CAS / SWP 写入内存的新值纯由 src 覆写，不混入旧内存污点；LDADD/STADD 等则与旧内存混合
                    let out_mem_tag = match akind {
                        AtomicKind::Cas { .. } | AtomicKind::Swp => src_tag,
                        AtomicKind::LdOp | AtomicKind::StOp => mem_tag | src_tag,
                    };
                    let mut dests = Vec::new();
                    for &d in &sem.dst_regs {
                        reg_taint[d as usize] = mem_tag;
                        if mem_tag != 0 {
                            dests.push(format_reg_slot(d));
                        }
                    }
                    let mut mem_written_tainted = false;
                    for w in &mem_writes {
                        let sz = w.size_or_reg.max(1) as usize;
                        let w_addr = w.addr;
                        set_mem_range_tag(&mut mem_taint, w_addr, sz, out_mem_tag);
                        if out_mem_tag != 0 {
                            mem_written_tainted = true;
                            dests.push(format!("mem[0x{w_addr:016x}+{sz}]"));
                        }
                    }
                    let step_mask = (if !sem.dst_regs.is_empty() { mem_tag } else { 0 })
                        | (if mem_written_tainted { out_mem_tag } else { 0 });
                    if step_mask != 0 && !dests.is_empty() {
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm,
                            kind: "rmw",
                            sources,
                            dests,
                            tag_mask: step_mask,
                            value: mem_writes.first().map(|w| w.value),
                        });
                    }
                }
                InsnCategory::AluRmw | InsnCategory::AluOverwrite => {
                    let mut tag = 0u64;
                    let mut sources = Vec::new();
                    for &r in &sem.src_regs {
                        let t = reg_taint[r as usize];
                        if t != 0 {
                            tag |= t;
                            sources.push(format_reg_slot(r));
                        }
                    }
                    let mut dests = Vec::new();
                    for &d in &sem.dst_regs {
                        reg_taint[d as usize] = tag;
                        if tag != 0 {
                            dests.push(format_reg_slot(d));
                        }
                    }
                    if tag != 0 && !dests.is_empty() {
                        let val = sem
                            .dst_regs
                            .first()
                            .and_then(|&d| lookup_reg_delta(self, idx, d));
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm,
                            kind: "alu",
                            sources,
                            dests,
                            tag_mask: tag,
                            value: val,
                        });
                    }
                }
            }

            // 应用附着在当前指令序号上的 HLE 批量拷贝/清零与新返回值事件（兼容 BL/BLR、.plt BR X17、尾调用 B 及跳板 RET）
            for hle in &hle_evts {
                apply_hle_forward(hle, idx, seq, &mut mem_taint, &mut push_step);
            }
            if hle_evts.iter().any(|h| hle_defines_fresh_x0(h.op)) {
                reg_taint[0] = 0;
            }
        }

        let tainted_regs: Vec<(String, u64)> = reg_taint
            .iter()
            .enumerate()
            .filter(|(_, &t)| t != 0)
            .map(|(slot, &t)| (format_reg_slot(slot as u8), t))
            .collect();
        let tainted_mem_ranges = coalesce_mem_tags(&mem_taint);

        TaintReport {
            mode: "forward",
            seed_desc: seed.describe(),
            anchor_idx: start_idx,
            stop_idx: end_idx,
            scanned_insns: end_idx.saturating_sub(start_idx),
            total_matching_steps,
            steps,
            tainted_regs,
            tainted_mem_ranges,
            origin_annotations: Vec::new(),
        }
    }

    /// 离线反向因果切片（Backward Causal Slicing）
    ///
    /// 从目标观测点 `opts.anchor_idx`（默认最后一条指令）的 `target`（例如某 8 字节密文或寄存器）
    /// 沿时间轴倒序回溯，精准追踪「到底是谁写、经由哪些指令与 HLE memcpy 算出来的」。
    /// 每次回溯命中覆盖写即立即 kill 该字节，绝不会被同一栈地址的历史/未来复用对象误导。
    pub fn taint_backward(&self, target: &TaintSeed, opts: &TaintOptions) -> TaintReport {
        let total = self.len();
        if total == 0 {
            return TaintReport {
                mode: "backward",
                seed_desc: target.describe(),
                anchor_idx: 0,
                stop_idx: 0,
                scanned_insns: 0,
                total_matching_steps: 0,
                steps: Vec::new(),
                tainted_regs: Vec::new(),
                tainted_mem_ranges: Vec::new(),
                origin_annotations: Vec::new(),
            };
        }

        let anchor_idx = opts
            .anchor_idx
            .unwrap_or_else(|| total.saturating_sub(1))
            .min(total.saturating_sub(1));
        let mut stop_idx = if opts.max_insns > 0 {
            anchor_idx.saturating_sub(opts.max_insns.saturating_sub(1))
        } else {
            0
        };
        if let Some(min_i) = opts.min_idx {
            if min_i > anchor_idx {
                return TaintReport {
                    mode: "backward",
                    seed_desc: target.describe(),
                    anchor_idx,
                    stop_idx: min_i,
                    scanned_insns: 0,
                    total_matching_steps: 0,
                    steps: Vec::new(),
                    tainted_regs: Vec::new(),
                    tainted_mem_ranges: Vec::new(),
                    origin_annotations: Vec::new(),
                };
            }
            stop_idx = stop_idx.max(min_i);
        }
        let max_steps = if opts.max_steps == 0 {
            usize::MAX
        } else {
            opts.max_steps
        };
        let track_sp = opts.track_sp || matches!(target, TaintSeed::Reg { reg_id: 31 });
        let is_stop_mem = |addr: u64| -> bool {
            opts.stop_mem_ranges
                .iter()
                .any(|&(s, l)| addr >= s && addr < s.saturating_add(l as u64))
        };

        let mut wanted_regs = [false; NUM_REG_SLOTS];
        let mut wanted_mem: BTreeSet<u64> = BTreeSet::new();
        let mut pinned_origin_mem: BTreeSet<u64> = BTreeSet::new();
        let mut origin_annotations: Vec<String> = Vec::new();

        match *target {
            TaintSeed::Mem { addr, len } => {
                for off in 0..len {
                    if let Some(a) = addr.checked_add(off as u64) {
                        wanted_mem.insert(a);
                    }
                }
            }
            TaintSeed::Reg { reg_id } => {
                if (reg_id as usize) < NUM_REG_SLOTS {
                    wanted_regs[reg_id as usize] = true;
                }
            }
        }

        let mut rev_steps: Vec<TaintStep> = Vec::new();
        let mut total_matching_steps = 0usize;
        let mut scanned = 0usize;
        let mut effective_stop_idx = stop_idx;

        let mut push_step = |step: TaintStep| {
            total_matching_steps += 1;
            if rev_steps.len() < max_steps {
                rev_steps.push(step);
            }
        };

        for idx in (stop_idx..=anchor_idx).rev() {
            if wanted_mem.is_empty() && !wanted_regs.iter().any(|&w| w) {
                effective_stop_idx = idx.saturating_add(1).min(anchor_idx);
                break;
            }
            let Some(rec) = self.get(idx) else {
                continue;
            };
            let pc = rec.pc;
            let hit_stop_pc = idx < anchor_idx && opts.stop_at_pcs.contains(&pc);
            scanned += 1;
            let opcode = rec.opcode;
            let seq = rec.seq_low32;

            // 1. 逆序先处理附着在本条指令上的 HLE 批量拷贝/清零/外部源与新返回值事件
            let hle_evts = self.hle_events_for(idx);
            for hle in hle_evts.iter().rev() {
                // 外部熵源/时钟源写入内存（如 getrandom / gettimeofday / clock_gettime / time(tloc)）
                if (is_hle_entropy_or_time_mem(hle.op) || hle.op == "time")
                    && hle.dst != 0
                    && hle.len > 0
                {
                    let end = hle.dst.saturating_add(hle.len);
                    let hit_addrs: Vec<u64> = wanted_mem.range(hle.dst..end).copied().collect();
                    if !hit_addrs.is_empty() {
                        for a in &hit_addrs {
                            wanted_mem.remove(a);
                        }
                        let note = format!(
                            "[ORIGIN: HLE {}() #{} @ 0x{:016x} -> mem[0x{:016x}+{}]]",
                            hle.op, seq, hle.pc, hle.dst, hle.len
                        );
                        origin_annotations.push(note.clone());
                        let (hle_asm, _) = decode_semantics(opcode, pc, track_sp);
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm: hle_asm,
                            kind: "hle_origin",
                            sources: vec![note],
                            dests: vec![format!("mem[0x{:016x}+{}]", hle.dst, hle.len)],
                            tag_mask: 0,
                            value: Some(hle.src),
                        });
                    }
                    continue;
                }
                apply_hle_backward(hle, idx, seq, &mut wanted_mem, &mut push_step);
                // 若 HLE memcpy 将需求搬运进 stop_mem_ranges，立即钉死为边界输入
                if !opts.stop_mem_ranges.is_empty() {
                    let to_pin: Vec<u64> = wanted_mem.iter().copied().filter(|&a| is_stop_mem(a)).collect();
                    for a in to_pin {
                        wanted_mem.remove(&a);
                        pinned_origin_mem.insert(a);
                    }
                }
            }
            if wanted_regs[0] {
                if let Some(hle) = hle_evts.iter().rev().find(|h| hle_defines_fresh_x0(h.op)) {
                    wanted_regs[0] = false;
                    let fallback = match hle.op {
                        "posix_memalign" | "free" | "bzero" => Some(0),
                        "rand" | "random" | "lrand48" | "arc4random" | "time" => Some(hle.src),
                        _ => Some(hle.dst),
                    };
                    let ret_val = lookup_reg_delta(self, idx, 0).or(fallback);
                    let (hle_asm, _) = decode_semantics(opcode, pc, track_sp);
                    let is_entropy = is_hle_entropy_or_time_reg(hle.op);
                    let src_label = if is_entropy {
                        let vstr = ret_val.map(|v| format!(" -> 0x{v:x}")).unwrap_or_default();
                        let note = format!("[ORIGIN: HLE {}() #{seq} @ 0x{:016x}{vstr}]", hle.op, hle.pc);
                        origin_annotations.push(note.clone());
                        note
                    } else {
                        format!("HLE {}(len={})", hle.op, hle.len)
                    };
                    push_step(TaintStep {
                        idx,
                        seq,
                        pc,
                        asm: hle_asm,
                        kind: if is_entropy { "hle_origin" } else { "hle_ret" },
                        sources: vec![src_label],
                        dests: vec!["X0".to_string()],
                        tag_mask: 0,
                        value: ret_val,
                    });
                }
            }

            if wanted_mem.is_empty() && !wanted_regs.iter().any(|&w| w) {
                effective_stop_idx = idx;
                break;
            }

            let (asm, sem) = decode_semantics(opcode, pc, track_sp);
            let mem_evts = self.mem_events_for(idx);
            let (mem_reads, mem_writes) = split_mem_events(&mem_evts);

            match sem.category {
                InsnCategory::NoDataFlow => {}
                InsnCategory::Call => {
                    if wanted_regs[30] {
                        wanted_regs[30] = false;
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm,
                            kind: "const",
                            sources: vec![format!("const(pc+4=0x{:x})", pc.wrapping_add(4))],
                            dests: vec!["X30".to_string()],
                            tag_mask: 0,
                            value: Some(pc.wrapping_add(4)),
                        });
                    }
                }
                InsnCategory::ConstDef => {
                    let hit_dsts: Vec<u8> = sem
                        .dst_regs
                        .iter()
                        .copied()
                        .filter(|&d| wanted_regs[d as usize])
                        .collect();
                    if !hit_dsts.is_empty() {
                        for &d in &hit_dsts {
                            wanted_regs[d as usize] = false;
                        }
                        let val = lookup_reg_delta(self, idx, hit_dsts[0]);
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm,
                            kind: "const",
                            sources: vec!["const".to_string()],
                            dests: hit_dsts.iter().map(|&d| format_reg_slot(d)).collect(),
                            tag_mask: 0,
                            value: val,
                        });
                    }
                }
                InsnCategory::Store {
                    is_pair,
                    has_status_reg,
                } => {
                    // 优先处理 `MemPostIdxReg([base, stride])` 写回
                    if let Some((base, stride)) = sem.post_idx_writeback {
                        if wanted_regs[base as usize] {
                            wanted_regs[stride as usize] = true;
                        }
                    }
                    if has_status_reg {
                        let hit_status: Vec<u8> = sem
                            .dst_regs
                            .iter()
                            .copied()
                            .filter(|&d| wanted_regs[d as usize])
                            .collect();
                        if !hit_status.is_empty() {
                            for &d in &hit_status {
                                wanted_regs[d as usize] = false;
                            }
                            let val = lookup_reg_delta(self, idx, hit_status[0]);
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm: asm.clone(),
                                kind: "const",
                                sources: vec!["exclusive_status".to_string()],
                                dests: hit_status.iter().map(|&d| format_reg_slot(d)).collect(),
                                tag_mask: 0,
                                value: val,
                            });
                        }
                    }
                    if is_pair {
                        let slot_writes =
                            partition_pair_mem_events(&mem_writes, sem.pair_elem_size);
                        let mut sources = Vec::new();
                        let mut dests = Vec::new();
                        let mut first_val = None;
                        for slot_i in 0..2 {
                            let mut slot_hit = false;
                            for &(a, sz, v) in &slot_writes[slot_i] {
                                let mut hit = false;
                                for off in 0..sz {
                                    if let Some(addr_b) = a.checked_add(off as u64) {
                                        if wanted_mem.remove(&addr_b) {
                                            hit = true;
                                        }
                                    }
                                }
                                if hit {
                                    slot_hit = true;
                                    dests.push(format!("mem[0x{a:016x}+{sz}]"));
                                    if first_val.is_none() {
                                        first_val = Some(v);
                                    }
                                }
                            }
                            if slot_hit {
                                if let Some(src) = sem.pair_regs[slot_i] {
                                    wanted_regs[src as usize] = true;
                                    let sname = format_reg_slot(src);
                                    if !sources.contains(&sname) {
                                        sources.push(sname);
                                    }
                                } else if !sources.contains(&"const(0/XZR)".to_string()) {
                                    sources.push("const(0/XZR)".to_string());
                                }
                            }
                        }
                        if !dests.is_empty() {
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "store",
                                sources,
                                dests,
                                tag_mask: 0,
                                value: first_val,
                            });
                        }
                    } else {
                        let mut dests = Vec::new();
                        let mut first_val = None;
                        for w in &mem_writes {
                            let sz = w.size_or_reg.max(1) as usize;
                            let w_addr = w.addr;
                            let w_val = w.value;
                            let mut hit = false;
                            for off in 0..sz {
                                if let Some(addr_b) = w_addr.checked_add(off as u64) {
                                    if wanted_mem.remove(&addr_b) {
                                        hit = true;
                                    }
                                }
                            }
                            if hit {
                                dests.push(format!("mem[0x{w_addr:016x}+{sz}]"));
                                if first_val.is_none() {
                                    first_val = Some(w_val);
                                }
                            }
                        }
                        if !dests.is_empty() {
                            let mut sources = Vec::new();
                            for &r in &sem.src_regs {
                                wanted_regs[r as usize] = true;
                                sources.push(format_reg_slot(r));
                            }
                            if sources.is_empty() {
                                sources.push("const(0/XZR)".to_string());
                            }
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "store",
                                sources,
                                dests,
                                tag_mask: 0,
                                value: first_val,
                            });
                        }
                    }
                }
                InsnCategory::Load {
                    is_pair,
                    is_rmw_lane,
                } => {
                    if let Some((base, stride)) = sem.post_idx_writeback {
                        if wanted_regs[base as usize] {
                            wanted_regs[stride as usize] = true;
                        }
                    }
                    if is_pair {
                        let slot_reads = partition_pair_mem_events(&mem_reads, sem.pair_elem_size);
                        let mut sources = Vec::new();
                        let mut dests = Vec::new();
                        let mut first_dst = None;
                        for slot_i in 0..2 {
                            let Some(dst) = sem.pair_regs[slot_i] else {
                                continue;
                            };
                            if wanted_regs[dst as usize] {
                                wanted_regs[dst as usize] = false;
                                dests.push(format_reg_slot(dst));
                                if first_dst.is_none() {
                                    first_dst = Some(dst);
                                }
                                for &(a, sz, _) in &slot_reads[slot_i] {
                                    for off in 0..sz {
                                        if let Some(addr_b) = a.checked_add(off as u64) {
                                            if is_stop_mem(addr_b) {
                                                pinned_origin_mem.insert(addr_b);
                                            } else {
                                                wanted_mem.insert(addr_b);
                                            }
                                        }
                                    }
                                    sources.push(format!("mem[0x{a:016x}+{sz}]"));
                                }
                            }
                        }
                        if !dests.is_empty() {
                            if opts.track_mem_index {
                                for &r in &sem.bwd_mem_index_regs {
                                    wanted_regs[r as usize] = true;
                                    sources.push(format_reg_slot(r));
                                }
                            }
                            let val = first_dst.and_then(|d| lookup_reg_delta(self, idx, d));
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "load",
                                sources,
                                dests,
                                tag_mask: 0,
                                value: val,
                            });
                        }
                    } else {
                        let hit_dsts: Vec<u8> = sem
                            .dst_regs
                            .iter()
                            .copied()
                            .filter(|&d| wanted_regs[d as usize])
                            .collect();
                        if !hit_dsts.is_empty() {
                            if !is_rmw_lane {
                                for &d in &hit_dsts {
                                    wanted_regs[d as usize] = false;
                                }
                            }
                            let mut sources = Vec::new();
                            for r in &mem_reads {
                                let sz = r.size_or_reg.max(1) as usize;
                                let r_addr = r.addr;
                                for off in 0..sz {
                                    if let Some(addr_b) = r_addr.checked_add(off as u64) {
                                        if is_stop_mem(addr_b) {
                                            pinned_origin_mem.insert(addr_b);
                                        } else {
                                            wanted_mem.insert(addr_b);
                                        }
                                    }
                                }
                                sources.push(format!("mem[0x{r_addr:016x}+{sz}]"));
                            }
                            if opts.track_mem_index {
                                for &r in &sem.bwd_mem_index_regs {
                                    wanted_regs[r as usize] = true;
                                    sources.push(format_reg_slot(r));
                                }
                            }
                            let val = lookup_reg_delta(self, idx, hit_dsts[0]);
                            push_step(TaintStep {
                                idx,
                                seq,
                                pc,
                                asm,
                                kind: "load",
                                sources,
                                dests: hit_dsts.iter().map(|&d| format_reg_slot(d)).collect(),
                                tag_mask: 0,
                                value: val,
                            });
                        }
                    }
                }
                InsnCategory::AtomicRmw(akind) => {
                    let mut dests = Vec::new();
                    let mut sources = Vec::new();
                    let mut hit_mem = false;
                    for w in &mem_writes {
                        let sz = w.size_or_reg.max(1) as usize;
                        let w_addr = w.addr;
                        let mut hit = false;
                        for off in 0..sz {
                            if let Some(addr_b) = w_addr.checked_add(off as u64) {
                                if wanted_mem.remove(&addr_b) {
                                    hit = true;
                                }
                            }
                        }
                        if hit {
                            hit_mem = true;
                            dests.push(format!("mem[0x{w_addr:016x}+{sz}]"));
                        }
                    }
                    let mut hit_reg = false;
                    for &d in &sem.dst_regs {
                        if wanted_regs[d as usize] {
                            wanted_regs[d as usize] = false;
                            hit_reg = true;
                            dests.push(format_reg_slot(d));
                        }
                    }
                    if hit_mem || hit_reg {
                        // CAS / SWP 仅当需要旧寄存器值时才回溯 mem_reads；LDADD/STADD 写内存也依赖旧内存值
                        let need_old_mem = match akind {
                            AtomicKind::Cas { .. } | AtomicKind::Swp => hit_reg,
                            AtomicKind::LdOp | AtomicKind::StOp => true,
                        };
                        if need_old_mem {
                            for r in &mem_reads {
                                let sz = r.size_or_reg.max(1) as usize;
                                let r_addr = r.addr;
                                for off in 0..sz {
                                    if let Some(addr_b) = r_addr.checked_add(off as u64) {
                                        if is_stop_mem(addr_b) {
                                            pinned_origin_mem.insert(addr_b);
                                        } else {
                                            wanted_mem.insert(addr_b);
                                        }
                                    }
                                }
                                sources.push(format!("mem[0x{r_addr:016x}+{sz}]"));
                            }
                        }
                        if hit_mem {
                            for &r in &sem.src_regs {
                                wanted_regs[r as usize] = true;
                                sources.push(format_reg_slot(r));
                            }
                        }
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm,
                            kind: "rmw",
                            sources,
                            dests,
                            tag_mask: 0,
                            value: mem_writes.first().map(|w| w.value),
                        });
                    }
                }
                InsnCategory::AluRmw | InsnCategory::AluOverwrite => {
                    let hit_dsts: Vec<u8> = sem
                        .dst_regs
                        .iter()
                        .copied()
                        .filter(|&d| wanted_regs[d as usize])
                        .collect();
                    if !hit_dsts.is_empty() {
                        if sem.category == InsnCategory::AluOverwrite {
                            for &d in &hit_dsts {
                                wanted_regs[d as usize] = false;
                            }
                        }
                        let mut sources = Vec::new();
                        for &r in &sem.src_regs {
                            wanted_regs[r as usize] = true;
                            sources.push(format_reg_slot(r));
                        }
                        if sources.is_empty() {
                            sources.push("const".to_string());
                        }
                        let val = lookup_reg_delta(self, idx, hit_dsts[0]);
                        push_step(TaintStep {
                            idx,
                            seq,
                            pc,
                            asm,
                            kind: "alu",
                            sources,
                            dests: hit_dsts.iter().map(|&d| format_reg_slot(d)).collect(),
                            tag_mask: 0,
                            value: val,
                        });
                    }
                }
            }
            if hit_stop_pc {
                effective_stop_idx = idx;
                break;
            }
        }

        // 反转为时间正序（起源 -> 目标），便于人类和 AI Agent 自上而下阅读
        rev_steps.reverse();
        origin_annotations.reverse();
        wanted_mem.extend(pinned_origin_mem);

        let tainted_regs: Vec<(String, u64)> = wanted_regs
            .iter()
            .enumerate()
            .filter(|(_, &w)| w)
            .map(|(slot, _)| (format_reg_slot(slot as u8), 0))
            .collect();
        let tainted_mem_ranges = coalesce_mem_set(&wanted_mem);

        TaintReport {
            mode: "backward",
            seed_desc: target.describe(),
            anchor_idx,
            stop_idx: effective_stop_idx,
            scanned_insns: scanned,
            total_matching_steps,
            steps: rev_steps,
            tainted_regs,
            tainted_mem_ranges,
            origin_annotations,
        }
    }
}

fn apply_hle_forward<F>(
    hle: &HleTraceEvent,
    idx: usize,
    seq: u32,
    mem_taint: &mut HashMap<u64, u64>,
    push_step: &mut F,
) where
    F: FnMut(TaintStep),
{
    let len = hle.len;
    if len == 0 {
        return;
    }
    match hle.op {
        "memcpy" | "memmove" | "realloc_copy" => {
            let src_last = hle.src.saturating_add(len - 1);
            let dst_last = hle.dst.saturating_add(len - 1);
            // 先快照源区间污点（O(|tainted|)，支持 memmove 重叠拷贝与 >4MiB 大块拷贝，且防 u64 溢出）
            let mut copied: Vec<(u64, u64)> = Vec::new();
            let mut combined = 0u64;
            for (&a, &tag) in mem_taint.iter() {
                if tag != 0 && a >= hle.src && a <= src_last {
                    let off = a - hle.src;
                    copied.push((off, tag));
                    combined |= tag;
                }
            }
            // 清除目标区间原有污点，再写入从源拷贝过来的污点
            mem_taint.retain(|&a, _| !(a >= hle.dst && a <= dst_last));
            for (off, tag) in &copied {
                if let Some(da) = hle.dst.checked_add(*off) {
                    mem_taint.insert(da, *tag);
                }
            }
            if combined != 0 {
                push_step(TaintStep {
                    idx,
                    seq,
                    pc: hle.pc,
                    asm: format!(
                        "HLE {}(dst=0x{:x}, src=0x{:x}, len={})",
                        hle.op, hle.dst, hle.src, hle.len
                    ),
                    kind: "hle_copy",
                    sources: vec![format!("mem[0x{:016x}+{}]", hle.src, hle.len)],
                    dests: vec![format!("mem[0x{:016x}+{}]", hle.dst, hle.len)],
                    tag_mask: combined,
                    value: Some(hle.len),
                });
            }
        }
        "memset" | "bzero" | "calloc_zero" => {
            let dst_last = hle.dst.saturating_add(len - 1);
            let mut cleared = false;
            mem_taint.retain(|&a, _| {
                if a >= hle.dst && a <= dst_last {
                    cleared = true;
                    false
                } else {
                    true
                }
            });
            if cleared {
                let fill_byte = hle.src as u8;
                push_step(TaintStep {
                    idx,
                    seq,
                    pc: hle.pc,
                    asm: format!(
                        "HLE {}(dst=0x{:x}, byte=0x{:02x}, len={})",
                        hle.op, hle.dst, fill_byte, hle.len
                    ),
                    kind: "hle_clear",
                    sources: vec![format!("const(0x{fill_byte:02x})")],
                    dests: vec![format!("mem[0x{:016x}+{}]", hle.dst, hle.len)],
                    tag_mask: 0,
                    value: Some(fill_byte as u64),
                });
            }
        }
        op if (is_hle_entropy_or_time_mem(op) || op == "time") && hle.dst != 0 => {
            let dst_last = hle.dst.saturating_add(len - 1);
            let mut cleared = false;
            mem_taint.retain(|&a, _| {
                if a >= hle.dst && a <= dst_last {
                    cleared = true;
                    false
                } else {
                    true
                }
            });
            if cleared {
                push_step(TaintStep {
                    idx,
                    seq,
                    pc: hle.pc,
                    asm: format!("HLE {}(dst=0x{:x}, len={})", hle.op, hle.dst, hle.len),
                    kind: "hle_clear",
                    sources: vec![format!("HLE {}()", hle.op)],
                    dests: vec![format!("mem[0x{:016x}+{}]", hle.dst, hle.len)],
                    tag_mask: 0,
                    value: Some(hle.src),
                });
            }
        }
        _ => {}
    }
}

fn apply_hle_backward<F>(
    hle: &HleTraceEvent,
    idx: usize,
    seq: u32,
    wanted_mem: &mut BTreeSet<u64>,
    push_step: &mut F,
) where
    F: FnMut(TaintStep),
{
    let len = hle.len;
    if len == 0 || wanted_mem.is_empty() {
        return;
    }
    match hle.op {
        "memcpy" | "memmove" | "realloc_copy" => {
            // 闭区间 `dst..=dst.saturating_add(len - 1)` 彻底杜绝高地址溢出触发 BTreeSet::range Panic
            let dst_last = hle.dst.saturating_add(len - 1);
            let hits: Vec<u64> = wanted_mem.range(hle.dst..=dst_last).copied().collect();
            if hits.is_empty() {
                return;
            }
            for &a in &hits {
                wanted_mem.remove(&a);
            }
            let mut mapped_src = BTreeSet::new();
            for &a in &hits {
                let off = a - hle.dst;
                if let Some(s) = hle.src.checked_add(off) {
                    wanted_mem.insert(s);
                    mapped_src.insert(s);
                }
            }
            let src_ranges: Vec<String> = coalesce_mem_set(&mapped_src)
                .into_iter()
                .map(|(a, l, _)| format!("mem[0x{a:016x}+{l}]"))
                .collect();
            let dst_set: BTreeSet<u64> = hits.into_iter().collect();
            let dst_ranges: Vec<String> = coalesce_mem_set(&dst_set)
                .into_iter()
                .map(|(a, l, _)| format!("mem[0x{a:016x}+{l}]"))
                .collect();
            push_step(TaintStep {
                idx,
                seq,
                pc: hle.pc,
                asm: format!(
                    "HLE {}(dst=0x{:x}, src=0x{:x}, len={})",
                    hle.op, hle.dst, hle.src, hle.len
                ),
                kind: "hle_copy",
                sources: src_ranges,
                dests: dst_ranges,
                tag_mask: 0,
                value: Some(hle.len),
            });
        }
        "memset" | "bzero" | "calloc_zero" => {
            let dst_last = hle.dst.saturating_add(len - 1);
            let hits: Vec<u64> = wanted_mem.range(hle.dst..=dst_last).copied().collect();
            if hits.is_empty() {
                return;
            }
            for &a in &hits {
                wanted_mem.remove(&a);
            }
            let dst_set: BTreeSet<u64> = hits.into_iter().collect();
            let dst_ranges: Vec<String> = coalesce_mem_set(&dst_set)
                .into_iter()
                .map(|(a, l, _)| format!("mem[0x{a:016x}+{l}]"))
                .collect();
            let fill_byte = hle.src as u8;
            push_step(TaintStep {
                idx,
                seq,
                pc: hle.pc,
                asm: format!(
                    "HLE {}(dst=0x{:x}, byte=0x{:02x}, len={})",
                    hle.op, hle.dst, fill_byte, hle.len
                ),
                kind: "hle_clear",
                sources: vec![format!("const(0x{fill_byte:02x})")],
                dests: dst_ranges,
                tag_mask: 0,
                value: Some(fill_byte as u64),
            });
        }
        _ => {}
    }
}
