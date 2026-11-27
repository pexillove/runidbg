//! ARM64 定制字节码虚拟机（VM / OLLVM 控制流平坦化）离线专项分析引擎
//!
//! 直接基于 `trace.idx` + `trace.ext` 实现零插桩开销的离线分析：
//! 1. **自动识别调度循环（Dispatch Loop）与跳转表（Jump Table）**：统计热间接分支 `BR Xn` / `BLR Xn`
//!    的分发目标集合，语义校验 8B 绝对表与 4B 相对偏移表（彻底免疫 4B 字节码取指误判），
//!    并用 `bad64` 反推加载指令基址寄存器得到精确跳转表基址；
//! 2. **自动推断字节码 PC 寄存器 / 内存槽位与指令步长**：识别每次分发单调步进的寄存器（如 `X23`）
//!    或内存单元（如 `[X23]`），统计每类 Handler 实际消费的字节数直方图（4/8/12/… 字节）；
//! 3. **自定义指令字位域解码器**：支持声明 `op=[lsb, width]` 与多个操作数字段，自动输出 `op=49 f=[1,5,0,7]`；
//! 4. **字节级虚拟寄存器堆符号化差分（即时感知基址搬家 + 跨帧分桶 + HLE 同步）**：
//!    给定 `baseReg`（如 `X25`）、`count`、`size`，即时响应 Handler 内 `baseReg` 搬家并按基址分桶保存影子帧，
//!    完整兼容窄写/高半写/宽写与 `HleMemcpy`/`HleMemset`。

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::tracer::reader::TraceReader;
use crate::tracer::record::{ExtKind, ExtRecord};
use crate::tracer::taint::{format_reg_slot, parse_reg_slot};

/// 位域定义：`[lsb, lsb + width - 1]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmBitField {
    pub lsb: u8,
    pub width: u8,
}

impl VmBitField {
    pub fn new(lsb: u8, width: u8) -> Option<Self> {
        if lsb >= 64 || width == 0 || (lsb as u16 + width as u16) > 64 {
            return None;
        }
        Some(Self { lsb, width })
    }

    #[inline]
    pub fn extract(&self, word: u64) -> u64 {
        let shifted = word >> self.lsb;
        if self.width >= 64 {
            shifted
        } else {
            shifted & ((1u64 << self.width) - 1)
        }
    }
}

/// 字节码指令字位域布局（如 `op=[6,6], fields=[[12,5],[17,5],[22,5],[27,5]]`）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmWordLayout {
    pub op: VmBitField,
    pub fields: Vec<VmBitField>,
}

/// 解码后的单条字节码指令字
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmDecodedWord {
    pub raw: u64,
    pub op: u64,
    pub fields: Vec<u64>,
}

impl VmDecodedWord {
    pub fn format_compact(&self) -> String {
        if self.fields.is_empty() {
            format!("op={}", self.op)
        } else {
            let fs: Vec<String> = self.fields.iter().map(|v| v.to_string()).collect();
            format!("op={} f=[{}]", self.op, fs.join(","))
        }
    }
}

impl VmWordLayout {
    pub fn decode(&self, word: u64) -> VmDecodedWord {
        let op = self.op.extract(word);
        let fields = self.fields.iter().map(|f| f.extract(word)).collect();
        VmDecodedWord { raw: word, op, fields }
    }

    /// 解析 CLI 布局字符串，格式：`<op_lsb>:<op_width>[,<f0_lsb>[:<f0_width>],...]`
    /// 例如 `"6:6,12:5,17:5,22:5,27:5"` 或省略字段宽度 `"6:6,12,17,22,27"`（自动按相邻差值推算，其余默认 `min(5, 64 - lsb)`）
    pub fn parse_spec(spec: &str) -> Result<Self, String> {
        let parts: Vec<&str> = spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err("layout 不能为空，示例: 6:6,12:5,17:5,22:5,27:5".to_string());
        }
        let parse_pair = |s: &str| -> Result<(u8, Option<u8>), String> {
            if let Some((a, b)) = s.split_once(':') {
                let lsb = a
                    .trim()
                    .parse::<u8>()
                    .map_err(|_| format!("无效位偏移: {a}"))?;
                let width = b
                    .trim()
                    .parse::<u8>()
                    .map_err(|_| format!("无效位宽: {b}"))?;
                Ok((lsb, Some(width)))
            } else {
                let lsb = s.parse::<u8>().map_err(|_| format!("无效位偏移: {s}"))?;
                Ok((lsb, None))
            }
        };

        let (op_lsb, op_w_opt) = parse_pair(parts[0])?;
        let op_w = op_w_opt.ok_or_else(|| "首项 op 必须指定宽度 <lsb>:<width>".to_string())?;
        let op = VmBitField::new(op_lsb, op_w)
            .ok_or_else(|| format!("op 位域越界: lsb={op_lsb}, width={op_w}"))?;

        let raw_fields: Vec<(u8, Option<u8>)> = parts[1..]
            .iter()
            .map(|p| parse_pair(p))
            .collect::<Result<_, _>>()?;
        let mut fields = Vec::with_capacity(raw_fields.len());
        for (i, &(lsb, w_opt)) in raw_fields.iter().enumerate() {
            let width = match w_opt {
                Some(w) => w,
                None => {
                    if let Some(&(next_lsb, _)) = raw_fields.get(i + 1) {
                        if next_lsb > lsb && (next_lsb - lsb) <= 16 {
                            next_lsb - lsb
                        } else {
                            5u8.min(64u8.saturating_sub(lsb))
                        }
                    } else {
                        5u8.min(64u8.saturating_sub(lsb))
                    }
                }
            };
            let bf = VmBitField::new(lsb, width)
                .ok_or_else(|| format!("字段位域越界: lsb={lsb}, width={width}"))?;
            fields.push(bf);
        }
        Ok(Self { op, fields })
    }
}

/// 离线虚拟寄存器堆追踪配置
#[derive(Debug, Clone)]
pub struct VmRegfileConfig {
    /// 基址寄存器槽位（0..=31，如 X25 = 25）
    pub base_reg: u8,
    /// 虚拟寄存器数量（默认 32，范围 1..=256）
    pub count: usize,
    /// 单个虚拟寄存器字节宽（4 或 8，默认 8）
    pub size: u8,
}

/// 离线 VM 分析配置
#[derive(Debug, Clone)]
pub struct VmAnalysisOptions {
    /// 手动指定调度点 PC（`None` 时自动选取最高分热间接分支）
    pub dispatcher_pc: Option<u64>,
    /// 手动指定字节码指针寄存器槽位（如 `X23` = 23；`None` 时自动推断）
    pub bc_reg: Option<u8>,
    /// 字节码指针是否存放在 `[bc_reg]` 内存中（即一级间接指针，须配合 `bc_reg` 使用）
    pub bc_deref: bool,
    /// 可选：直接指定存放字节码指针的内存地址（`bc_deref` 的定址形式，与 `bc_reg` 互斥）
    pub bc_ptr_addr: Option<u64>,
    /// 可选：虚拟寄存器堆符号化差分配置
    pub regfile: Option<VmRegfileConfig>,
    /// 可选：字节码指令字位域解码器
    pub layout: Option<VmWordLayout>,
    /// 识别调度点所需最少不同 handler 目标数（默认 3）
    pub min_handlers: usize,
    /// 识别调度点所需最少执行次数（默认 4）
    pub min_dispatches: usize,
    /// 可选：起始指令下标（含；等价 `--seq-range <start>..<end>`）
    pub min_idx: Option<usize>,
    /// 可选：结束指令下标（含；等价 `--seq-range <start>..<end>`）
    pub max_idx: Option<usize>,
    /// 最多扫描的原生指令条数（0 = 全部）
    pub max_insns: usize,
    /// 报告中最多保留的详细 VM 步数（0 = 全部，默认 256）
    pub max_steps: usize,
}

impl Default for VmAnalysisOptions {
    fn default() -> Self {
        Self {
            dispatcher_pc: None,
            bc_reg: None,
            bc_deref: false,
            bc_ptr_addr: None,
            regfile: None,
            layout: None,
            min_handlers: 3,
            min_dispatches: 4,
            min_idx: None,
            max_idx: None,
            max_insns: 0,
            max_steps: 256,
        }
    }
}

/// 自动发现的 VM 调度循环候选点
#[derive(Debug, Clone)]
pub struct VmDispatcherCandidate {
    pub pc: u64,
    pub kind: &'static str, // "br" | "blr"
    pub target_reg: u8,
    pub total_dispatches: usize,
    pub unique_handlers: usize,
    pub jump_table_base: Option<u64>,
    pub jump_table_entry_size: Option<u8>,
    pub jump_table_load_pc: Option<u64>,
    pub detected_bc_reg: Option<u8>,
    pub detected_bc_mem_addr: Option<u64>,
    pub score: u64,
}

/// 单个 VM Handler 的聚合画像（对应自动化生成的 `vm_opmap`）
#[derive(Debug, Clone)]
pub struct VmHandlerStat {
    pub handler_pc: u64,
    pub opcodes: Vec<u64>,
    pub hit_count: usize,
    pub min_arm64_insns: usize,
    pub max_arm64_insns: usize,
    pub total_arm64_insns: u64,
    /// 字节码指针步长直方图：`step_bytes -> 出现次数`（如 `{4: 80, 8: 12}`）
    pub bc_step_hist: BTreeMap<i64, usize>,
    pub first_step: usize,
    pub first_trace_idx: usize,
}

impl VmHandlerStat {
    pub fn avg_arm64_insns(&self) -> f64 {
        if self.hit_count == 0 {
            0.0
        } else {
            self.total_arm64_insns as f64 / self.hit_count as f64
        }
    }
}

/// 单步 VM 虚拟寄存器差分
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmRegDiff {
    pub reg_index: usize,
    pub old_val: Option<u64>,
    pub new_val: u64,
}

impl VmRegDiff {
    pub fn format_compact(&self) -> String {
        match self.old_val {
            Some(o) => format!("R[{}]: 0x{:x}->0x{:x}", self.reg_index, o, self.new_val),
            None => format!("R[{}]: ?->0x{:x}", self.reg_index, self.new_val),
        }
    }
}

/// 单条恢复出的 VM 字节码执行步
#[derive(Debug, Clone)]
pub struct VmStep {
    pub step: usize,
    pub trace_idx: usize,
    pub seq: u32,
    pub dispatcher_pc: u64,
    pub handler_pc: u64,
    pub arm64_insns: usize,
    pub bc_pc: Option<u64>,
    pub bc_step: Option<i64>,
    pub raw_word: Option<u64>,
    pub decoded: Option<VmDecodedWord>,
    pub table_opcode: Option<u64>,
    pub regfile_base: Option<u64>,
    pub base_relocated: bool,
    pub reg_diffs: Vec<VmRegDiff>,
}

/// 完整离线 VM 分析报告
#[derive(Debug, Clone)]
pub struct VmAnalysisReport {
    pub scanned_insns: usize,
    pub dispatchers: Vec<VmDispatcherCandidate>,
    pub selected_dispatcher: Option<u64>,
    pub detected_bc_reg: Option<String>,
    pub detected_bc_mem_addr: Option<u64>,
    pub total_vm_steps: usize,
    pub handlers: Vec<VmHandlerStat>,
    pub steps: Vec<VmStep>,
}

impl VmAnalysisReport {
    /// 格式化为人类与 AI Agent 一眼可读的 VM 逆向画像与 `vm_opmap` 表格
    pub fn format_summary(&self, base_addr: u64) -> String {
        let fmt_pc = |pc: u64| -> String {
            if base_addr != 0 && pc >= base_addr {
                format!("0x{pc:016x} (+0x{:x})", pc - base_addr)
            } else {
                format!("0x{pc:016x}")
            }
        };

        let mut out = String::new();
        out.push_str(&format!(
            "# VM 离线专项分析: 扫描原生指令={} | 候选调度点={} | 选中调度点={} | 总 VM 步数={} | 唯一 Handler={}\n",
            self.scanned_insns,
            self.dispatchers.len(),
            self.selected_dispatcher
                .map(fmt_pc)
                .unwrap_or_else(|| "无".to_string()),
            self.total_vm_steps,
            self.handlers.len()
        ));
        if let Some(ref reg) = self.detected_bc_reg {
            out.push_str(&format!("# 字节码 PC 寄存器: {reg}\n"));
        } else if let Some(addr) = self.detected_bc_mem_addr {
            out.push_str(&format!("# 字节码 PC 内存槽位: 0x{addr:016x}\n"));
        }

        if !self.dispatchers.is_empty() {
            out.push_str("\n## 1. 候选调度循环（Dispatch Loop & Jump Table）\n");
            for (i, d) in self.dispatchers.iter().take(8).enumerate() {
                let tbl = match (d.jump_table_base, d.jump_table_entry_size) {
                    (Some(b), Some(sz)) => format!(" table=0x{b:x} (entry={}B)", sz),
                    (Some(b), None) => format!(" table=0x{b:x}"),
                    _ => String::new(),
                };
                let bc = if let Some(r) = d.detected_bc_reg {
                    format!(" bc_reg={}", format_reg_slot(r))
                } else if let Some(a) = d.detected_bc_mem_addr {
                    format!(" bc_mem=0x{a:x}")
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "  [{}] {} ({} {}) | hits={} | handlers={} | score={}{}{}\n",
                    i,
                    fmt_pc(d.pc),
                    d.kind,
                    format_reg_slot(d.target_reg),
                    d.total_dispatches,
                    d.unique_handlers,
                    d.score,
                    tbl,
                    bc
                ));
            }
        }

        if !self.handlers.is_empty() {
            out.push_str("\n## 2. Handler 分布图（vm_opmap & 字节码步长统计）\n");
            for h in &self.handlers {
                let ops = if h.opcodes.is_empty() {
                    "-".to_string()
                } else {
                    let s: Vec<String> = h.opcodes.iter().take(6).map(|v| v.to_string()).collect();
                    if h.opcodes.len() > 6 {
                        format!("[{},..+{}]", s.join(","), h.opcodes.len() - 6)
                    } else {
                        format!("[{}]", s.join(","))
                    }
                };
                let steps_str = if h.bc_step_hist.is_empty() {
                    "-".to_string()
                } else {
                    let parts: Vec<String> = h
                        .bc_step_hist
                        .iter()
                        .map(|(st, cnt)| format!("{st:+}B×{cnt}"))
                        .collect();
                    parts.join(", ")
                };
                out.push_str(&format!(
                    "  handler={} | op={} | hits={:<5} | arm64={:.1} ({}..{}) | bc_step=[{}] | first=#{}\n",
                    fmt_pc(h.handler_pc),
                    ops,
                    h.hit_count,
                    h.avg_arm64_insns(),
                    h.min_arm64_insns,
                    h.max_arm64_insns,
                    steps_str,
                    h.first_trace_idx
                ));
            }
        }

        if !self.steps.is_empty() {
            out.push_str(&format!(
                "\n## 3. VM 执行轨迹（前 {} / {} 步）\n",
                self.steps.len(),
                self.total_vm_steps
            ));
            for st in &self.steps {
                let bc_part = match (st.bc_pc, st.bc_step) {
                    (Some(p), Some(d)) => format!(" bc=0x{p:x}({d:+}B)"),
                    (Some(p), None) => format!(" bc=0x{p:x}"),
                    _ => String::new(),
                };
                let op_part = if let Some(ref dec) = st.decoded {
                    format!(" {}", dec.format_compact())
                } else if let Some(op) = st.table_opcode {
                    format!(" op={op}")
                } else if let Some(w) = st.raw_word {
                    format!(" word=0x{w:x}")
                } else {
                    String::new()
                };
                let reloc = if st.base_relocated {
                    format!(
                        " [REGFILE->0x{:x}]",
                        st.regfile_base.unwrap_or(0)
                    )
                } else {
                    String::new()
                };
                let diffs = if st.reg_diffs.is_empty() {
                    String::new()
                } else {
                    let ds: Vec<String> =
                        st.reg_diffs.iter().map(VmRegDiff::format_compact).collect();
                    format!(" | {}", ds.join(", "))
                };
                out.push_str(&format!(
                    "  step#{:04} idx=#{:06} h={} ({} insns){}{}{}{}\n",
                    st.step,
                    st.trace_idx,
                    fmt_pc(st.handler_pc),
                    st.arm64_insns,
                    bc_part,
                    op_part,
                    reloc,
                    diffs
                ));
            }
        }

        out
    }
}

/// 字节级虚拟寄存器影子槽（完整支持 1B/2B/4B/8B 窄写、高半写、宽写与 HLE memset/memcpy）
#[derive(Debug, Clone, Copy, Default)]
struct ShadowSlot {
    bytes: [u8; 8],
    known_mask: u8,
}

impl ShadowSlot {
    #[inline]
    fn value_opt(&self, slot_size: u8) -> Option<u64> {
        let valid_bits = if slot_size == 4 { 0x0Fu8 } else { 0xFFu8 };
        if (self.known_mask & valid_bits) == 0 {
            return None;
        }
        let raw = u64::from_le_bytes(self.bytes);
        Some(if slot_size == 4 {
            raw & 0xFFFF_FFFF
        } else {
            raw
        })
    }

    #[inline]
    fn write_byte(&mut self, idx: usize, val: u8) {
        if idx < 8 {
            self.bytes[idx] = val;
            self.known_mask |= 1u8 << idx;
        }
    }

    #[inline]
    fn clear_byte(&mut self, idx: usize) {
        if idx < 8 {
            self.bytes[idx] = 0;
            self.known_mask &= !(1u8 << idx);
        }
    }
}

/// 快速识别 ARM64 间接跳转指令：`BR Xn` 或 `BLR Xn`
#[inline]
fn decode_indirect_branch(opcode: u32) -> Option<(&'static str, u8)> {
    // BR Xn:  1101 0110 0001 1111 0000 00nn nnn0 0000 -> 0xD61F_0000, mask 0xFFFF_FC1F
    // BLR Xn: 1101 0110 0011 1111 0000 00nn nnn0 0000 -> 0xD63F_0000, mask 0xFFFF_FC1F
    let masked = opcode & 0xFFFF_FC1F;
    let rn = ((opcode >> 5) & 0x1F) as u8;
    if masked == 0xD61F_0000 {
        Some(("br", rn))
    } else if masked == 0xD63F_0000 {
        Some(("blr", rn))
    } else {
        None
    }
}

/// 用 `bad64` 提取加载指令的基址寄存器槽位（用于定位真实跳转表基址寄存器 `Rn`）
fn extract_load_base_reg(opcode: u32, pc: u64) -> Option<u8> {
    let insn = bad64::decode(opcode, pc).ok()?;
    for op in insn.operands() {
        let reg = match *op {
            bad64::Operand::MemReg(r)
            | bad64::Operand::MemOffset { reg: r, .. }
            | bad64::Operand::MemPreIdx { reg: r, .. }
            | bad64::Operand::MemPostIdxReg([r, _])
            | bad64::Operand::MemPostIdxImm { reg: r, .. }
            | bad64::Operand::MemExt { regs: [r, _], .. } => Some(r),
            _ => None,
        };
        if let Some(r) = reg {
            let name = format!("{r}");
            if let Some(slot) = parse_gpr_slot(&name) {
                return Some(slot);
            }
        }
    }
    None
}

/// 便捷解析寄存器名（复用 taint 模块映射，仅允许通用寄存器 X0..X30 / SP）
pub fn parse_gpr_slot(name: &str) -> Option<u8> {
    parse_reg_slot(name).filter(|&slot| slot <= 31)
}

/// 应用字节级写入区间到指定基址的影子寄存器堆，并记录本步 `(old_val, new_val)` 差分
#[allow(clippy::too_many_arguments)]
fn apply_regfile_bytes_write(
    shadow_rfs: &mut HashMap<u64, Vec<ShadowSlot>>,
    base: u64,
    cfg: &VmRegfileConfig,
    write_addr: u64,
    data: &[u8],
    record_diff: bool,
    cur_step_rf_writes: &mut BTreeMap<usize, (Option<u64>, u64)>,
) {
    let span = (cfg.count as u64).saturating_mul(cfg.size as u64);
    let rf_end = base.saturating_add(span);
    let w_end = write_addr.saturating_add(data.len() as u64);
    let ov_start = write_addr.max(base);
    let ov_end = w_end.min(rf_end);
    if ov_start >= ov_end {
        return;
    }

    let slots = shadow_rfs
        .entry(base)
        .or_insert_with(|| vec![ShadowSlot::default(); cfg.count]);
    let slot_sz = cfg.size as u64;
    let first_slot = ((ov_start - base) / slot_sz) as usize;
    let last_slot = (((ov_end - 1) - base) / slot_sz) as usize;

    for (slot_idx, slot) in slots
        .iter_mut()
        .enumerate()
        .take(last_slot.min(cfg.count - 1) + 1)
        .skip(first_slot)
    {
        let old_val = slot.value_opt(cfg.size);
        let slot_start = base + (slot_idx as u64) * slot_sz;
        let slot_end = slot_start + slot_sz;
        let b_start = ov_start.max(slot_start);
        let b_end = ov_end.min(slot_end);
        for a in b_start..b_end {
            let src_i = (a - write_addr) as usize;
            let dst_b = (a - slot_start) as usize;
            slot.write_byte(dst_b, data[src_i]);
        }
        if record_diff {
            if let Some(new_val) = slot.value_opt(cfg.size) {
                cur_step_rf_writes
                    .entry(slot_idx)
                    .and_modify(|e| e.1 = new_val)
                    .or_insert((old_val, new_val));
            }
        }
    }
}

/// 应用 HLE `memset`/`bzero`/`calloc_zero` 到当前影子寄存器堆
#[allow(clippy::too_many_arguments)]
fn apply_regfile_memset(
    shadow_rfs: &mut HashMap<u64, Vec<ShadowSlot>>,
    base: u64,
    cfg: &VmRegfileConfig,
    dst: u64,
    len: u64,
    fill_byte: u8,
    record_diff: bool,
    cur_step_rf_writes: &mut BTreeMap<usize, (Option<u64>, u64)>,
) {
    if len == 0 {
        return;
    }
    let span = (cfg.count as u64).saturating_mul(cfg.size as u64);
    let rf_end = base.saturating_add(span);
    let w_end = dst.saturating_add(len);
    let ov_start = dst.max(base);
    let ov_end = w_end.min(rf_end);
    if ov_start >= ov_end {
        return;
    }

    let slots = shadow_rfs
        .entry(base)
        .or_insert_with(|| vec![ShadowSlot::default(); cfg.count]);
    let slot_sz = cfg.size as u64;
    let first_slot = ((ov_start - base) / slot_sz) as usize;
    let last_slot = (((ov_end - 1) - base) / slot_sz) as usize;

    for (slot_idx, slot) in slots
        .iter_mut()
        .enumerate()
        .take(last_slot.min(cfg.count - 1) + 1)
        .skip(first_slot)
    {
        let old_val = slot.value_opt(cfg.size);
        let slot_start = base + (slot_idx as u64) * slot_sz;
        let slot_end = slot_start + slot_sz;
        let b_start = ov_start.max(slot_start);
        let b_end = ov_end.min(slot_end);
        for a in b_start..b_end {
            slot.write_byte((a - slot_start) as usize, fill_byte);
        }
        if record_diff {
            if let Some(new_val) = slot.value_opt(cfg.size) {
                cur_step_rf_writes
                    .entry(slot_idx)
                    .and_modify(|e| e.1 = new_val)
                    .or_insert((old_val, new_val));
            }
        }
    }
}

/// 应用 HLE `memcpy`/`memmove`/`realloc_copy` 到当前影子寄存器堆（跨帧/帧内拷贝保留已知字节，外部源置未知）
#[allow(clippy::too_many_arguments)]
fn apply_regfile_memcpy(
    shadow_rfs: &mut HashMap<u64, Vec<ShadowSlot>>,
    base: u64,
    cfg: &VmRegfileConfig,
    dst: u64,
    src: u64,
    len: u64,
    record_diff: bool,
    cur_step_rf_writes: &mut BTreeMap<usize, (Option<u64>, u64)>,
) {
    if len == 0 {
        return;
    }
    let span = (cfg.count as u64).saturating_mul(cfg.size as u64);
    let rf_end = base.saturating_add(span);
    let w_end = dst.saturating_add(len);
    let ov_start = dst.max(base);
    let ov_end = w_end.min(rf_end);
    if ov_start >= ov_end {
        return;
    }
    let copy_n = (ov_end - ov_start) as usize;
    let slot_sz = cfg.size as u64;

    // 先从已有影子帧快照源字节（支持重叠 memmove 与跨帧拷贝！）
    let mut src_snapshot: Vec<Option<u8>> = vec![None; copy_n];
    for (i, out_b) in src_snapshot.iter_mut().enumerate() {
        let src_a = src.wrapping_add(ov_start - dst).wrapping_add(i as u64);
        for (&fb, fslots) in shadow_rfs.iter() {
            if src_a >= fb && src_a < fb.saturating_add(span) {
                let s_idx = ((src_a - fb) / slot_sz) as usize;
                let b_idx = ((src_a - fb) % slot_sz) as usize;
                if let Some(sl) = fslots.get(s_idx) {
                    if (sl.known_mask & (1u8 << b_idx)) != 0 {
                        *out_b = Some(sl.bytes[b_idx]);
                    }
                }
                break;
            }
        }
    }

    let slots = shadow_rfs
        .entry(base)
        .or_insert_with(|| vec![ShadowSlot::default(); cfg.count]);
    let first_slot = ((ov_start - base) / slot_sz) as usize;
    let last_slot = (((ov_end - 1) - base) / slot_sz) as usize;
    for (slot_idx, slot) in slots
        .iter_mut()
        .enumerate()
        .take(last_slot.min(cfg.count - 1) + 1)
        .skip(first_slot)
    {
        let old_val = slot.value_opt(cfg.size);
        let slot_start = base + (slot_idx as u64) * slot_sz;
        let slot_end = slot_start + slot_sz;
        let b_start = ov_start.max(slot_start);
        let b_end = ov_end.min(slot_end);
        for a in b_start..b_end {
            let snap_i = (a - ov_start) as usize;
            let dst_b = (a - slot_start) as usize;
            match src_snapshot[snap_i] {
                Some(v) => slot.write_byte(dst_b, v),
                None => slot.clear_byte(dst_b),
            }
        }
        if record_diff {
            if let Some(new_val) = slot.value_opt(cfg.size) {
                cur_step_rf_writes
                    .entry(slot_idx)
                    .and_modify(|e| e.1 = new_val)
                    .or_insert((old_val, new_val));
            }
        }
    }
}

/// 跳转表探测结果
struct JumpTableInfo {
    base: u64,
    entry_size: u8,
    load_pc: u64,
    /// 若通过 `bad64` 解出加载基址寄存器，Phase 2 可直接读取其运行时值作为真实表基址
    base_reg: Option<u8>,
}

impl TraceReader {
    /// 离线执行 VM 专项分析（自动识别调度循环、跳转表、Handler 分布、字节码步长与虚拟寄存器堆差分）
    pub fn analyze_vm(&self, opts: &VmAnalysisOptions) -> VmAnalysisReport {
        let total = self.len();
        let scan_start = opts.min_idx.unwrap_or(0).min(total);
        let mut scan_end = opts
            .max_idx
            .map(|m| m.saturating_add(1).min(total))
            .unwrap_or(total);
        if opts.max_insns > 0 {
            scan_end = scan_end.min(scan_start.saturating_add(opts.max_insns));
        }
        let scan_len = scan_end;
        let window_insns = scan_end.saturating_sub(scan_start);
        let safe_bc_reg = opts.bc_reg.filter(|&r| r < 32);
        let safe_regfile = opts
            .regfile
            .clone()
            .filter(|c| c.base_reg < 32)
            .map(|mut c| {
                c.count = c.count.clamp(1, 256);
                c.size = if c.size == 4 { 4 } else { 8 };
                c
            });

        if window_insns < 2 {
            return VmAnalysisReport {
                scanned_insns: window_insns,
                dispatchers: Vec::new(),
                selected_dispatcher: opts.dispatcher_pc,
                detected_bc_reg: safe_bc_reg.map(format_reg_slot),
                detected_bc_mem_addr: opts.bc_ptr_addr,
                total_vm_steps: 0,
                handlers: Vec::new(),
                steps: Vec::new(),
            };
        }

        // ── Phase 1: 单遍扫描窗口内所有间接分支 BR / BLR，按分数初筛 Top-8 候选再做深度探测 ──
        struct RawDisp {
            kind: &'static str,
            target_reg: u8,
            hit_indices: Vec<usize>,
            targets: BTreeSet<u64>,
        }
        let mut raw_disps: HashMap<u64, RawDisp> = HashMap::new();
        for idx in scan_start..scan_end.saturating_sub(1) {
            let Some(rec) = self.get(idx) else { continue };
            let Some((kind, rn)) = decode_indirect_branch(rec.opcode) else {
                continue;
            };
            let Some(next) = self.get(idx + 1) else {
                continue;
            };
            // 校验 seq_low32 连续性，防止因 mmap 丢帧把断层后的无关指令 PC 误当跳转目标
            if next.seq_low32 != rec.seq_low32.wrapping_add(1) {
                continue;
            }
            let pc = rec.pc;
            let next_pc = next.pc;
            let entry = raw_disps.entry(pc).or_insert_with(|| RawDisp {
                kind,
                target_reg: rn,
                hit_indices: Vec::new(),
                targets: BTreeSet::new(),
            });
            entry.hit_indices.push(idx);
            entry.targets.insert(next_pc);
        }

        let min_h = opts.min_handlers.max(1);
        let min_d = opts.min_dispatches.max(1);
        let mut scored_pcs: Vec<(u64, u64)> = Vec::new();
        for (&pc, rd) in &raw_disps {
            let forced = opts.dispatcher_pc == Some(pc);
            if !forced && (rd.targets.len() < min_h || rd.hit_indices.len() < min_d) {
                continue;
            }
            let kind_bonus: u64 = if rd.kind == "br" { 2 } else { 1 };
            let score = (rd.hit_indices.len() as u64)
                .saturating_mul(rd.targets.len() as u64)
                .saturating_mul(kind_bonus);
            scored_pcs.push((score, pc));
        }
        scored_pcs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        // 仅对 Top-8 候选（及显式指定 dispatcher）执行跳转表与字节码 PC 深度启发式
        let mut top_pcs: Vec<(u64, u64)> = scored_pcs.into_iter().take(8).collect();
        if let Some(forced_pc) = opts.dispatcher_pc {
            if let Some(rd) = raw_disps.get(&forced_pc) {
                if !top_pcs.iter().any(|&(_, p)| p == forced_pc) {
                    let kind_bonus: u64 = if rd.kind == "br" { 2 } else { 1 };
                    let score = (rd.hit_indices.len() as u64)
                        .saturating_mul(rd.targets.len() as u64)
                        .saturating_mul(kind_bonus);
                    top_pcs.push((score, forced_pc));
                }
            }
        }

        let mut candidates: Vec<VmDispatcherCandidate> = Vec::with_capacity(top_pcs.len());
        let mut jtbl_map: HashMap<u64, JumpTableInfo> = HashMap::new();
        for (score, pc) in top_pcs {
            let rd = &raw_disps[&pc];
            let jtbl = self.detect_jump_table(&rd.hit_indices, &rd.targets);
            let (det_bc_reg, det_bc_mem) = self.detect_bytecode_pc(&rd.hit_indices);
            candidates.push(VmDispatcherCandidate {
                pc,
                kind: rd.kind,
                target_reg: rd.target_reg,
                total_dispatches: rd.hit_indices.len(),
                unique_handlers: rd.targets.len(),
                jump_table_base: jtbl.as_ref().map(|j| j.base),
                jump_table_entry_size: jtbl.as_ref().map(|j| j.entry_size),
                jump_table_load_pc: jtbl.as_ref().map(|j| j.load_pc),
                detected_bc_reg: det_bc_reg,
                detected_bc_mem_addr: det_bc_mem,
                score,
            });
            if let Some(j) = jtbl {
                jtbl_map.insert(pc, j);
            }
        }

        let selected_pc = opts
            .dispatcher_pc
            .or_else(|| candidates.first().map(|c| c.pc));

        let Some(disp_pc) = selected_pc else {
            return VmAnalysisReport {
                scanned_insns: window_insns,
                dispatchers: candidates,
                selected_dispatcher: None,
                detected_bc_reg: safe_bc_reg.map(format_reg_slot),
                detected_bc_mem_addr: opts.bc_ptr_addr,
                total_vm_steps: 0,
                handlers: Vec::new(),
                steps: Vec::new(),
            };
        };

        // 收集选中调度点的全部命中下标（支持用户手动指定任意非 BR 指令作为调度点）
        let hit_indices: Vec<usize> = if let Some(rd) = raw_disps.get(&disp_pc) {
            rd.hit_indices.clone()
        } else {
            let mut v = Vec::new();
            for idx in scan_start..scan_end.saturating_sub(1) {
                if let (Some(rec), Some(next)) = (self.get(idx), self.get(idx + 1)) {
                    if rec.pc == disp_pc && next.seq_low32 == rec.seq_low32.wrapping_add(1) {
                        v.push(idx);
                    }
                }
            }
            v
        };

        if hit_indices.is_empty() {
            return VmAnalysisReport {
                scanned_insns: window_insns,
                dispatchers: candidates,
                selected_dispatcher: Some(disp_pc),
                detected_bc_reg: safe_bc_reg.map(format_reg_slot),
                detected_bc_mem_addr: opts.bc_ptr_addr,
                total_vm_steps: 0,
                handlers: Vec::new(),
                steps: Vec::new(),
            };
        }

        let selected_cand = candidates.iter().find(|c| c.pc == disp_pc).cloned();
        let selected_jtbl = jtbl_map.remove(&disp_pc);

        // 确定字节码 PC 来源（显式参数优先；bc_deref 仅在显式提供 bc_reg 时生效，绝不误套到自动推断出的直址寄存器）
        let effective_bc_deref = opts.bc_deref && safe_bc_reg.is_some();
        let (active_bc_reg, active_bc_mem) = if opts.bc_ptr_addr.is_some() {
            (None, opts.bc_ptr_addr)
        } else if let Some(r) = safe_bc_reg {
            (Some(r), None)
        } else if let Some(ref c) = selected_cand {
            (c.detected_bc_reg, c.detected_bc_mem_addr)
        } else {
            self.detect_bytecode_pc(&hit_indices)
        };

        // ── Phase 2: 单遍时间轴前向追踪寄存器状态 + 字节码指针 + 字节级分桶虚拟寄存器堆差分 ──
        let mut live_regs: [Option<u64>; 32] = [None; 32];
        let mut live_bc_mem_val: Option<u64> = None;

        // 按基址分桶的虚拟寄存器堆影子状态（完美隔离跨帧写入并支持嵌套 CALL/RET 返回原帧！）
        let mut shadow_rfs: HashMap<u64, Vec<ShadowSlot>> = HashMap::new();
        let mut current_rf_base: Option<u64> = None;

        let num_steps = hit_indices.len();
        let mut step_bc_pcs: Vec<Option<u64>> = vec![None; num_steps];
        let mut step_raw_words: Vec<Option<u64>> = vec![None; num_steps];
        let mut step_table_ops: Vec<Option<u64>> = vec![None; num_steps];
        let mut step_rf_bases: Vec<Option<u64>> = vec![None; num_steps];
        let mut step_rf_relocs: Vec<bool> = vec![false; num_steps];
        let mut step_rf_diffs: Vec<Vec<VmRegDiff>> = vec![Vec::new(); num_steps];
        let mut step_handler_pcs: Vec<u64> = vec![0; num_steps];
        let mut step_seqs: Vec<u32> = vec![0; num_steps];
        let mut step_arm_counts: Vec<usize> = vec![0; num_steps];

        let mut disp_cursor = 0usize;
        let mut recent_reads: Vec<(usize, ExtRecord)> = Vec::with_capacity(64);
        let mut active_step: Option<usize> = None;
        let mut cur_step_rf_writes: BTreeMap<usize, (Option<u64>, u64)> = BTreeMap::new();
        let mut cur_step_relocated = false;
        let mut cur_step_end_rf_base: Option<u64> = None;
        let mut refined_jtbl_base: Option<u64> = selected_jtbl.as_ref().map(|j| j.base);

        let last_needed_idx = scan_len.saturating_sub(1);
        for idx in 0..=last_needed_idx {
            // 1. 先处理当前指令的内存事件（取指、bc_mem 8B 指针读写、字节级 regfile 写）
            let mems = self.mem_events_for(idx);
            for m in &mems {
                let kind = ExtKind::from_u8(m.kind);
                let m_addr = m.addr;
                let m_val = m.value;
                let m_sz = m.size_or_reg;
                if matches!(kind, Some(ExtKind::MemRead)) {
                    recent_reads.push((idx, *m));
                    if recent_reads.len() > 64 {
                        recent_reads.drain(0..32);
                    }
                    // 若开启 bc_deref 且读取 8B 指针命中 [bc_reg]，更新字节码 PC
                    if m_sz == 8 {
                        if effective_bc_deref {
                            if let Some(br) = active_bc_reg {
                                if live_regs.get(br as usize).copied().flatten() == Some(m_addr) {
                                    live_bc_mem_val = Some(m_val);
                                }
                            }
                        }
                        if active_bc_mem == Some(m_addr) {
                            live_bc_mem_val = Some(m_val);
                        }
                    }
                } else if matches!(kind, Some(ExtKind::MemWrite)) {
                    if m_sz == 8 {
                        if effective_bc_deref {
                            if let Some(br) = active_bc_reg {
                                if live_regs.get(br as usize).copied().flatten() == Some(m_addr) {
                                    live_bc_mem_val = Some(m_val);
                                }
                            }
                        }
                        if active_bc_mem == Some(m_addr) {
                            live_bc_mem_val = Some(m_val);
                        }
                    }
                    // 字节级追踪虚拟寄存器堆写（支持 1/2/4/8B 窄写、高半偏移与双槽宽写）
                    if let (Some(cfg), Some(base)) = (safe_regfile.as_ref(), current_rf_base) {
                        let n = (m_sz as usize).clamp(1, 8);
                        let raw_bytes = m_val.to_le_bytes();
                        apply_regfile_bytes_write(
                            &mut shadow_rfs,
                            base,
                            cfg,
                            m_addr,
                            &raw_bytes[..n],
                            active_step.is_some(),
                            &mut cur_step_rf_writes,
                        );
                    }
                }
            }

            // 2. 处理附着在当前指令上的 HLE 批量内存事件（memcpy/memmove/memset/bzero）
            let hles = self.hle_events_for(idx);
            if !hles.is_empty() {
                for hle in &hles {
                    match hle.op {
                        "memset" | "bzero" | "calloc_zero" => {
                            let fill = (hle.src & 0xFF) as u8;
                            if let Some(baddr) = active_bc_mem {
                                if hle.len > 0
                                    && baddr >= hle.dst
                                    && baddr < hle.dst.saturating_add(hle.len)
                                {
                                    live_bc_mem_val = Some(u64::from_le_bytes([fill; 8]));
                                }
                            }
                            if let (Some(cfg), Some(base)) = (safe_regfile.as_ref(), current_rf_base) {
                                apply_regfile_memset(
                                    &mut shadow_rfs,
                                    base,
                                    cfg,
                                    hle.dst,
                                    hle.len,
                                    fill,
                                    active_step.is_some(),
                                    &mut cur_step_rf_writes,
                                );
                            }
                        }
                        "memcpy" | "memmove" | "realloc_copy" => {
                            if let Some(baddr) = active_bc_mem {
                                if hle.len > 0
                                    && baddr >= hle.dst
                                    && baddr < hle.dst.saturating_add(hle.len)
                                {
                                    live_bc_mem_val = None;
                                }
                            }
                            if let (Some(cfg), Some(base)) = (safe_regfile.as_ref(), current_rf_base) {
                                apply_regfile_memcpy(
                                    &mut shadow_rfs,
                                    base,
                                    cfg,
                                    hle.dst,
                                    hle.src,
                                    hle.len,
                                    active_step.is_some(),
                                    &mut cur_step_rf_writes,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }

            // 3. 若命中跳转表加载指令且已解出基址寄存器，细化真实跳转表基址（消除未命中 opcode 0 时的偏移错位）
            if let Some(ref jt) = selected_jtbl {
                if let (Some(rec), Some(br)) = (self.get(idx), jt.base_reg) {
                    if rec.pc == jt.load_pc {
                        if let Some(real_base) = live_regs.get(br as usize).copied().flatten() {
                            if real_base >= 0x1000 && real_base <= jt.base {
                                refined_jtbl_base = Some(real_base);
                            }
                        }
                    }
                }
            }

            // 4. 检查是否到达下一个 dispatch 点
            if disp_cursor < num_steps && hit_indices[disp_cursor] == idx {
                // 结算上一步的 regfile diff 与基址搬家标记
                if let Some(prev_s) = active_step {
                    let diffs: Vec<VmRegDiff> = cur_step_rf_writes
                        .iter()
                        .filter(|(_, (o, n))| *o != Some(*n))
                        .map(|(&reg_index, &(old_val, new_val))| VmRegDiff {
                            reg_index,
                            old_val,
                            new_val,
                        })
                        .collect();
                    step_rf_diffs[prev_s] = diffs;
                    if cur_step_relocated {
                        step_rf_relocs[prev_s] = true;
                        if let Some(nb) = cur_step_end_rf_base {
                            step_rf_bases[prev_s] = Some(nb);
                        }
                    }
                    cur_step_rf_writes.clear();
                    cur_step_relocated = false;
                    cur_step_end_rf_base = None;
                }

                let cur_s = disp_cursor;
                disp_cursor += 1;
                active_step = Some(cur_s);

                if let Some(rec) = self.get(idx) {
                    step_seqs[cur_s] = rec.seq_low32;
                    if let Some(next_rec) = self.get(idx + 1) {
                        if next_rec.seq_low32 == rec.seq_low32.wrapping_add(1) {
                            step_handler_pcs[cur_s] = next_rec.pc;
                        }
                    }
                }
                let next_disp_idx = if cur_s + 1 < num_steps {
                    hit_indices[cur_s + 1]
                } else {
                    scan_len
                };
                step_arm_counts[cur_s] = next_disp_idx.saturating_sub(idx);

                // 记录当前步的字节码 PC 与虚拟寄存器堆基址
                let cur_bc = if effective_bc_deref || active_bc_mem.is_some() {
                    live_bc_mem_val
                } else {
                    active_bc_reg.and_then(|r| live_regs.get(r as usize).copied().flatten())
                };
                step_bc_pcs[cur_s] = cur_bc;
                step_rf_bases[cur_s] = current_rf_base;

                // 取指与跳转表匹配窗口严格截断在上一次 dispatch 之后，避免紧凑循环+后增 PC 误取上一步旧字！
                let prev_disp_bound = if cur_s > 0 {
                    hit_indices[cur_s - 1].saturating_add(1)
                } else {
                    0
                };
                let win_start = idx.saturating_sub(16).max(prev_disp_bound);

                if let Some(bc) = cur_bc {
                    let mut best_word: Option<(u64, u64)> = None; // (distance, value)
                    for &(r_idx, r_ext) in recent_reads.iter().rev() {
                        if r_idx < win_start {
                            break;
                        }
                        let ra = r_ext.addr;
                        let rv = r_ext.value;
                        if ra == bc {
                            best_word = Some((0, rv));
                            break;
                        }
                        if bc >= ra && bc - ra <= 8 {
                            let dist = bc - ra;
                            if best_word.is_none_or(|(d, _)| dist < d) {
                                best_word = Some((dist, rv));
                            }
                        }
                    }
                    step_raw_words[cur_s] = best_word.map(|(_, v)| v);
                }

                // 仅接受 load_pc 精确匹配的跳转表读取事件，彻底杜绝误匹配同段数据读
                if let (Some(jt), Some(tb)) = (selected_jtbl.as_ref(), refined_jtbl_base) {
                    let esz = jt.entry_size;
                    let esz_u64 = esz as u64;
                    for &(r_idx, r_ext) in recent_reads.iter().rev() {
                        if r_idx < win_start {
                            break;
                        }
                        if r_ext.pc == jt.load_pc
                            && r_ext.size_or_reg == esz
                            && r_ext.addr >= tb
                            && (r_ext.addr - tb).is_multiple_of(esz_u64)
                        {
                            let op_idx = (r_ext.addr - tb) / esz_u64;
                            if op_idx < 4096 {
                                step_table_ops[cur_s] = Some(op_idx);
                                break;
                            }
                        }
                    }
                }
            }

            // 5. 应用当前指令执行后的寄存器增量，并即时响应虚拟寄存器堆基址搬家！
            for (rid, val) in self.reg_deltas_for(idx) {
                if (rid as usize) < 32 {
                    live_regs[rid as usize] = Some(val);
                    if let Some(ref cfg) = safe_regfile {
                        if rid == cfg.base_reg {
                            if let Some(old_base) = current_rf_base {
                                if old_base != val && active_step.is_some() {
                                    // 即时切换至新基址桶并标记本步搬家，彻底隔离旧帧写入与新帧写入！
                                    cur_step_relocated = true;
                                    cur_step_end_rf_base = Some(val);
                                    cur_step_rf_writes.clear();
                                }
                            }
                            current_rf_base = Some(val);
                        }
                    }
                }
            }
        }

        // 结算最后一步的 regfile diff 与基址搬家标记
        if let Some(last_s) = active_step {
            let diffs: Vec<VmRegDiff> = cur_step_rf_writes
                .iter()
                .filter(|(_, (o, n))| *o != Some(*n))
                .map(|(&reg_index, &(old_val, new_val))| VmRegDiff {
                    reg_index,
                    old_val,
                    new_val,
                })
                .collect();
            step_rf_diffs[last_s] = diffs;
            if cur_step_relocated {
                step_rf_relocs[last_s] = true;
                if let Some(nb) = cur_step_end_rf_base {
                    step_rf_bases[last_s] = Some(nb);
                }
            }
        }

        // 若通过基址寄存器细化了真实跳转表基址，同步回 candidates 展示
        if let Some(rb) = refined_jtbl_base {
            if let Some(c) = candidates.iter_mut().find(|c| c.pc == disp_pc) {
                c.jump_table_base = Some(rb);
            }
        }

        // ── Phase 3: 组装每步 VmStep（计算 bc_step 与指令字解码）并聚合 Handler 分布图 ──
        let mut handler_map: BTreeMap<u64, VmHandlerStat> = BTreeMap::new();
        let mut handler_ops: HashMap<u64, BTreeSet<u64>> = HashMap::new();
        let max_steps = if opts.max_steps == 0 {
            usize::MAX
        } else {
            opts.max_steps
        };
        let mut steps = Vec::with_capacity(num_steps.min(max_steps));

        for s in 0..num_steps {
            let bc_pc = step_bc_pcs[s];
            let bc_step = match (bc_pc, step_bc_pcs.get(s + 1).copied().flatten()) {
                (Some(a), Some(b)) => {
                    let diff = (b as i128) - (a as i128);
                    if (-4096..=4096).contains(&diff) {
                        Some(diff as i64)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let raw_word = step_raw_words[s];
            let decoded = match (raw_word, opts.layout.as_ref()) {
                (Some(w), Some(lay)) => Some(lay.decode(w)),
                _ => None,
            };
            let table_opcode = step_table_ops[s];
            let handler_pc = step_handler_pcs[s];
            let arm_insns = step_arm_counts[s];

            let hstat = handler_map.entry(handler_pc).or_insert_with(|| VmHandlerStat {
                handler_pc,
                opcodes: Vec::new(),
                hit_count: 0,
                min_arm64_insns: usize::MAX,
                max_arm64_insns: 0,
                total_arm64_insns: 0,
                bc_step_hist: BTreeMap::new(),
                first_step: s,
                first_trace_idx: hit_indices[s],
            });
            hstat.hit_count += 1;
            hstat.min_arm64_insns = hstat.min_arm64_insns.min(arm_insns);
            hstat.max_arm64_insns = hstat.max_arm64_insns.max(arm_insns);
            hstat.total_arm64_insns = hstat.total_arm64_insns.saturating_add(arm_insns as u64);
            if let Some(st) = bc_step {
                *hstat.bc_step_hist.entry(st).or_insert(0) += 1;
            }
            if let Some(ref d) = decoded {
                handler_ops.entry(handler_pc).or_default().insert(d.op);
            } else if let Some(op) = table_opcode {
                handler_ops.entry(handler_pc).or_default().insert(op);
            }

            if steps.len() < max_steps {
                steps.push(VmStep {
                    step: s,
                    trace_idx: hit_indices[s],
                    seq: step_seqs[s],
                    dispatcher_pc: disp_pc,
                    handler_pc,
                    arm64_insns: arm_insns,
                    bc_pc,
                    bc_step,
                    raw_word,
                    decoded,
                    table_opcode,
                    regfile_base: step_rf_bases[s],
                    base_relocated: step_rf_relocs[s],
                    reg_diffs: std::mem::take(&mut step_rf_diffs[s]),
                });
            }
        }

        let mut handlers: Vec<VmHandlerStat> = handler_map
            .into_values()
            .map(|mut h| {
                if let Some(ops) = handler_ops.remove(&h.handler_pc) {
                    h.opcodes = ops.into_iter().collect();
                }
                if h.min_arm64_insns == usize::MAX {
                    h.min_arm64_insns = 0;
                }
                h
            })
            .collect();
        handlers.sort_by(|a, b| {
            b.hit_count
                .cmp(&a.hit_count)
                .then_with(|| a.handler_pc.cmp(&b.handler_pc))
        });

        VmAnalysisReport {
            scanned_insns: window_insns,
            dispatchers: candidates,
            selected_dispatcher: Some(disp_pc),
            detected_bc_reg: active_bc_reg.map(format_reg_slot),
            detected_bc_mem_addr: active_bc_mem,
            total_vm_steps: num_steps,
            handlers,
            steps,
        }
    }

    /// 语义级定位跳转表（严格校验 8B 绝对目标匹配或 4B 常量基址相对偏移关系，绝不把 4B 字节码取指误判为跳转表）
    fn detect_jump_table(
        &self,
        hit_indices: &[usize],
        targets: &BTreeSet<u64>,
    ) -> Option<JumpTableInfo> {
        if self.ext_len() == 0 || hit_indices.is_empty() {
            return None;
        }
        let sample_n = hit_indices.len().min(64);
        // (entry_addr, loaded_val, target_pc, load_opcode)
        type JumpTableSample = (u64, u64, u64, u32);
        let mut by_pc_size: HashMap<(u64, u8), Vec<JumpTableSample>> = HashMap::new();
        for &h_idx in &hit_indices[..sample_n] {
            let Some(cur) = self.get(h_idx) else { continue };
            let Some(next) = self.get(h_idx + 1) else {
                continue;
            };
            if next.seq_low32 != cur.seq_low32.wrapping_add(1) {
                continue;
            }
            let target_pc = next.pc;
            let win_lo = h_idx.saturating_sub(8);
            for j in (win_lo..h_idx).rev() {
                let load_op = self.get(j).map(|r| r.opcode).unwrap_or(0);
                for m in self.mem_events_for(j) {
                    if m.kind != ExtKind::MemRead as u8 {
                        continue;
                    }
                    let sz = m.size_or_reg;
                    if sz != 4 && sz != 8 {
                        continue;
                    }
                    let m_addr = m.addr;
                    let m_val = m.value;
                    let m_pc = m.pc;
                    if sz == 8 && (m_val == target_pc || targets.contains(&m_val)) {
                        by_pc_size
                            .entry((m_pc, 8))
                            .or_default()
                            .push((m_addr, m_val, target_pc, load_op));
                        break;
                    }
                    if sz == 4 && (m_addr & 3) == 0 {
                        by_pc_size
                            .entry((m_pc, 4))
                            .or_default()
                            .push((m_addr, m_val, target_pc, load_op));
                        break;
                    }
                }
            }
        }

        let mut best: Option<(usize, JumpTableInfo)> = None;
        for ((load_pc, sz), samples) in by_pc_size {
            if samples.len() < 3 {
                continue;
            }
            // 1. 同一表项地址必须始终映射到同一 target_pc（同一地址对应不同目标说明不是静态跳转表）
            let mut addr_to_target: HashMap<u64, u64> = HashMap::new();
            let mut consistent = true;
            for &(addr, _, tpc, _) in &samples {
                if let Some(&prev_tpc) = addr_to_target.get(&addr) {
                    if prev_tpc != tpc {
                        consistent = false;
                        break;
                    }
                } else {
                    addr_to_target.insert(addr, tpc);
                }
            }
            if !consistent || addr_to_target.len() < 2 {
                continue;
            }

            // 2. 4B 候选必须满足常量相对基址语义（排除 4B 字节码取指 `ldr w8, [x23]`！）
            if sz == 4 {
                let first = samples[0];
                let rel_s1 = |v: u64, t: u64| t.wrapping_sub((v as u32 as i32 as i64) as u64);
                let rel_s4 =
                    |v: u64, t: u64| t.wrapping_sub(((v as u32 as i32 as i64) << 2) as u64);
                let rel_u1 = |v: u64, t: u64| t.wrapping_sub(v & 0xFFFF_FFFF);

                let b_s1 = rel_s1(first.1, first.2);
                let b_s4 = rel_s4(first.1, first.2);
                let b_u1 = rel_u1(first.1, first.2);

                let valid_rel = samples.iter().all(|&(_, v, t, _)| rel_s1(v, t) == b_s1)
                    || samples.iter().all(|&(_, v, t, _)| rel_s4(v, t) == b_s4)
                    || samples.iter().all(|&(_, v, t, _)| rel_u1(v, t) == b_u1);
                if !valid_rel {
                    continue;
                }
            }

            let mut addrs: Vec<u64> = addr_to_target.keys().copied().collect();
            addrs.sort_unstable();
            let min_a = addrs[0];
            let max_a = *addrs.last().unwrap();
            if max_a.saturating_sub(min_a) <= 32768
                && addrs
                    .iter()
                    .all(|&a| (a - min_a).is_multiple_of(sz as u64))
            {
                let base_reg = extract_load_base_reg(samples[0].3, load_pc);
                let score = samples.len() * if sz == 8 { 2 } else { 1 };
                if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                    best = Some((
                        score,
                        JumpTableInfo {
                            base: min_a,
                            entry_size: sz,
                            load_pc,
                            base_reg,
                        },
                    ));
                }
            }
        }
        best.map(|(_, info)| info)
    }

    /// 启发式推断字节码 PC 所在寄存器（如 `X23`）或内存槽位（如 `[sp+0x18]`）
    fn detect_bytecode_pc(&self, hit_indices: &[usize]) -> (Option<u8>, Option<u64>) {
        if hit_indices.len() < 3 {
            return (None, None);
        }
        let sample_n = hit_indices.len().min(64);
        let first_hit = hit_indices[0];
        let max_scan = hit_indices[sample_n - 1];
        let mut live_regs: [Option<u64>; 32] = [None; 32];
        let mut reg_samples: [Vec<u64>; 32] = std::array::from_fn(|_| Vec::new());
        let mut reg_fetch_hits: [usize; 32] = [0; 32];
        let mut mem_write_vals: HashMap<u64, Vec<u64>> = HashMap::new();

        let mut cur_hit = 0usize;
        for idx in 0..=max_scan {
            // 仅在进入 VM 调度之后采集 8B 内存写，防止 VM 初始化前的无关写占满 128 槽
            if idx >= first_hit {
                for m in self.mem_events_for(idx) {
                    if m.kind == ExtKind::MemWrite as u8
                        && m.size_or_reg == 8
                        && m.value >= 0x10000
                    {
                        let v = mem_write_vals.entry(m.addr).or_default();
                        if v.len() < 128 {
                            v.push(m.value);
                        }
                    } else if m.kind == ExtKind::MemRead as u8 && m.addr >= 0x10000 {
                        // 记录哪些寄存器恰好被用作取指读地址（平局时强优先于临时寄存器）
                        for r in 0..29usize {
                            if live_regs[r] == Some(m.addr) {
                                reg_fetch_hits[r] += 1;
                            }
                        }
                    }
                }
            }
            if cur_hit < sample_n && hit_indices[cur_hit] == idx {
                for r in 0..29usize {
                    if let Some(val) = live_regs[r] {
                        if val >= 0x10000 {
                            reg_samples[r].push(val);
                        }
                    }
                }
                cur_hit += 1;
            }
            for (rid, val) in self.reg_deltas_for(idx) {
                if (rid as usize) < 32 {
                    live_regs[rid as usize] = Some(val);
                }
            }
        }

        let score_seq = |vals: &[u64]| -> usize {
            if vals.len() < 3 {
                return 0;
            }
            let mut small_pos = 0usize;
            let mut changes = 0usize;
            for w in vals.windows(2) {
                if w[1] != w[0] {
                    changes += 1;
                    if w[1] > w[0] && (w[1] - w[0]) <= 64 {
                        small_pos += 1;
                    }
                }
            }
            if changes == 0 || small_pos * 2 < changes {
                return 0;
            }
            small_pos
        };

        // 综合评分：(步进得分, 作为读地址命中次数, 是否 callee-saved X19..X28)
        let mut best_reg: Option<((usize, usize, u8), u8)> = None;
        for (r, samples) in reg_samples.iter().enumerate().take(29) {
            let sc = score_seq(samples);
            if sc >= 2 {
                let callee_saved = u8::from((19..=28).contains(&r));
                let key = (sc, reg_fetch_hits[r], callee_saved);
                if best_reg.is_none_or(|(bk, _)| key > bk) {
                    best_reg = Some((key, r as u8));
                }
            }
        }
        if let Some((_, r)) = best_reg {
            return (Some(r), None);
        }

        let mut best_mem: Option<(usize, u64)> = None;
        for (addr, vals) in &mem_write_vals {
            let sc = score_seq(vals);
            if sc >= 2 && best_mem.is_none_or(|(bs, _)| sc > bs) {
                best_mem = Some((sc, *addr));
            }
        }
        (None, best_mem.map(|(_, a)| a))
    }
}
