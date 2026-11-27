use std::collections::HashMap;

use unicorn_engine::unicorn_const::Prot;
use unicorn_engine::{RegisterARM64, UcHookId, Unicorn};

use crate::backend::{BackendWithData, CpuBackend, ARM64_SVC_HLE_INSN};

use crate::error::EmuResult;
use crate::memory::{
    align_down, align_up, read_guest_cstr, read_guest_cstr_bytes, read_guest_u64, MemoryManager,
    PAGE_SIZE,
};
use crate::os::vfs::VirtualFs;

/// HLE 跳板区基址与大小（1MB，每槽 4 字节，最多容纳 262,144 个跳板入口）
pub const TRAMPOLINE_BASE: u64 = 0x1000_0000;
pub const TRAMPOLINE_SIZE: u64 = 0x0010_0000;

/// ARM64 `RET` 指令机器码（所有跳板槽位均填充此指令，Hook 执行完原生逻辑后直接返回调用方）
const ARM64_RET_INSN: u32 = 0xD65F_03C0;

/// 文件型 `mmap` 映射元数据，用于 `msync` / `munmap` 将脏页写回 `VirtualFs`
#[derive(Debug, Clone)]
pub(crate) struct MmapFileBacking {
    pub path: String,
    pub offset: usize,
    pub len: usize,
}

/// 内置高频 libc / Bionic / pthread / zlib HLE 劫持动作
#[derive(Debug, Clone)]
pub enum HleBuiltin {
    // 堆分配
    Malloc,
    Calloc,
    Realloc,
    PosixMemalign,
    Free,
    // 内存操作
    Memcpy,
    Memmove,
    Memset,
    Memcmp,
    Memchr,
    Bzero,
    // 字符串操作
    Strlen,
    Strnlen,
    Strcmp,
    Strncmp,
    Strcasecmp,
    Strncasecmp,
    Strcpy,
    Strncpy,
    Strcat,
    Strchr,
    Strrchr,
    Strstr,
    Strspn,
    Strcspn,
    Strdup,
    Strerror,
    // 格式化与转换
    Snprintf,
    SnprintfChk,
    Sprintf,
    SprintfChk,
    Asprintf,
    Vasprintf,
    Sscanf,
    Vsnprintf,
    VsnprintfChk,
    Atoi,
    Atol,
    Strtol,
    // POSIX 文件与内存映射（由 VirtualFs 支撑）
    Open,
    Close,
    Read,
    Write,
    Pread,
    Pwrite,
    Lseek,
    Fstat,
    Stat,
    Ftruncate,
    Access,
    Mkdir,
    Unlink,
    Rename,
    Mmap,
    Munmap,
    Msync,
    // Android / 系统 / zlib 杂项
    SystemPropertyGet,
    AndroidLogPrint,
    Errno,
    Getauxval,
    Getpagesize,
    Sysconf,
    Crc32,
    Arc4Random,
    Arc4RandomBuf,
    Rand,
    Srand,
    Getpid,
    Gettid,
    Getuid,
    Time,
    ClockGettime,
    Gettimeofday,
    LocaltimeR,
    Sleep,
    Usleep,
    Syscall,
    CxaAtexit,
    NoopZero,
    StackChkFail,
    Abort,
    Exit,
    Memalign,
    Printf,
    Fprintf,
    Ctype(&'static str),
    MathUnaryF64(&'static str),
    MathBinaryF64(&'static str),
    // Pthread 协作式线程与 TLS
    PthreadSelf,
    PthreadOnce,
    PthreadKeyCreate,
    PthreadSetSpecific,
    PthreadGetSpecific,
    PthreadCreate,
    PthreadJoin,
    PthreadDetach,
    PthreadExit,
    // 非局部跳转与信号处理
    Setjmp,
    Longjmp,
    Sigaction,
    Signal,
    Sigprocmask,
    Sigemptyset,
    Sigfillset,
    Sigaddset,
    Sigdelset,
    Sigismember,
    // Android NDK AAssetManager (libandroid.so)
    AAssetManagerFromJava,
    AAssetManagerOpen,
    AAssetManagerOpenDir,
    AAssetRead,
    AAssetSeek,
    AAssetGetLength,
    AAssetGetRemainingLength,
    AAssetGetBuffer,
    AAssetClose,
    AAssetDirGetNextFileName,
    AAssetDirRewind,
    AAssetDirClose,
    // zlib (libz.so)
    Adler32,
    CompressBound,
    Compress,
    Compress2,
    Uncompress,
    DeflateInit,
    DeflateInit2,
    Deflate,
    DeflateEnd,
    DeflateReset,
    InflateInit,
    InflateInit2,
    Inflate,
    InflateEnd,
    InflateReset,
    // POSIX Socket / EventLoop / DNS / 虚拟网卡 HLE
    Socket,
    Bind,
    Listen,
    Accept,
    Connect,
    Send,
    Sendto,
    Recv,
    Recvfrom,
    Shutdown,
    Setsockopt,
    Getsockopt,
    Getsockname,
    Getpeername,
    Poll,
    EpollCreate1,
    EpollCtl,
    EpollWait,
    Eventfd,
    Pipe,
    Getaddrinfo,
    Freeaddrinfo,
    InetPton,
    InetNtop,
    Htons,
    Htonl,
    Getifaddrs,
    Freeifaddrs,
    // C stdio (FILE*) 完整 HLE（直连 VirtualFs FD 表）
    Fopen,
    Fdopen,
    Freopen,
    Fclose,
    Fread,
    Fwrite,
    Fgets,
    FgetsChk,
    Fputs,
    Fgetc,
    Fputc,
    Ungetc,
    Fseek,
    Ftell,
    Rewind,
    Feof,
    Ferror,
    Clearerr,
    Fflush,
    Fileno,
    Fscanf,
    Getline,
    Getdelim,
    Lrand48,
    Mrand48,
    Srand48,
    Drand48,
    // 动态链接器运行时 API
    Dlopen,
    Dlsym,
    Dlclose,
    Dlerror,
    Dladdr,
    DlIteratePhdr,
    // 目录遍历 API
    Opendir,
    Readdir,
    Closedir,
    Rewinddir,
    // Bionic 新版系统属性 API
    SystemPropertyFind,
    SystemPropertyRead,
    SystemPropertyReadCallback,
    // 未实现的外部符号占位桩（触发时打印告警并返回 0，避免直接跳空指针崩溃）
    UnimplementedStub(String),
}

impl HleBuiltin {
    /// 按标准 C / Bionic 符号名匹配内置 HLE 实现
    pub fn from_symbol(name: &str) -> Option<Self> {
        match name {
            "malloc" => Some(Self::Malloc),
            "calloc" => Some(Self::Calloc),
            "realloc" => Some(Self::Realloc),
            "posix_memalign" => Some(Self::PosixMemalign),
            "free" => Some(Self::Free),
            "memcpy" | "__memcpy_chk" => Some(Self::Memcpy),
            "memmove" | "__memmove_chk" => Some(Self::Memmove),
            "memset" | "__memset_chk" => Some(Self::Memset),
            "memcmp" => Some(Self::Memcmp),
            "memchr" => Some(Self::Memchr),
            "bzero" => Some(Self::Bzero),
            "strlen" | "__strlen_chk" => Some(Self::Strlen),
            "strnlen" => Some(Self::Strnlen),
            "strcmp" => Some(Self::Strcmp),
            "strncmp" => Some(Self::Strncmp),
            "strcasecmp" => Some(Self::Strcasecmp),
            "strncasecmp" => Some(Self::Strncasecmp),
            "strcpy" | "__strcpy_chk" => Some(Self::Strcpy),
            "strncpy" | "__strncpy_chk" => Some(Self::Strncpy),
            "strcat" | "__strcat_chk" => Some(Self::Strcat),
            "strchr" | "__strchr_chk" => Some(Self::Strchr),
            "strrchr" | "__strrchr_chk" => Some(Self::Strrchr),
            "strstr" => Some(Self::Strstr),
            "strspn" => Some(Self::Strspn),
            "strcspn" => Some(Self::Strcspn),
            "strdup" => Some(Self::Strdup),
            "strerror" => Some(Self::Strerror),
            "snprintf" => Some(Self::Snprintf),
            "__snprintf_chk" => Some(Self::SnprintfChk),
            "sprintf" => Some(Self::Sprintf),
            "__sprintf_chk" => Some(Self::SprintfChk),
            "asprintf" => Some(Self::Asprintf),
            "vasprintf" => Some(Self::Vasprintf),
            "sscanf" | "__isoc99_sscanf" => Some(Self::Sscanf),
            "vsnprintf" => Some(Self::Vsnprintf),
            "__vsnprintf_chk" => Some(Self::VsnprintfChk),
            "atoi" => Some(Self::Atoi),
            "atol" | "atoll" => Some(Self::Atol),
            "strtol" | "strtoul" | "strtoll" | "strtoull" => Some(Self::Strtol),
            "open" | "__open_2" => Some(Self::Open),
            "close" => Some(Self::Close),
            "read" | "__read_chk" => Some(Self::Read),
            "write" => Some(Self::Write),
            "pread" | "pread64" | "__pread_chk" | "__pread64_chk" => Some(Self::Pread),
            "pwrite" | "pwrite64" => Some(Self::Pwrite),
            "lseek" | "lseek64" => Some(Self::Lseek),
            "fstat" | "fstat64" => Some(Self::Fstat),
            "stat" | "stat64" | "lstat" | "lstat64" => Some(Self::Stat),
            "ftruncate" | "ftruncate64" => Some(Self::Ftruncate),
            "access" => Some(Self::Access),
            "mkdir" => Some(Self::Mkdir),
            "unlink" => Some(Self::Unlink),
            "rename" => Some(Self::Rename),
            "mmap" | "mmap64" => Some(Self::Mmap),
            "munmap" => Some(Self::Munmap),
            "msync" => Some(Self::Msync),
            "__system_property_get" => Some(Self::SystemPropertyGet),
            "__android_log_print" | "__android_log_write" => Some(Self::AndroidLogPrint),
            "__errno" => Some(Self::Errno),
            "getauxval" => Some(Self::Getauxval),
            "getpagesize" => Some(Self::Getpagesize),
            "sysconf" => Some(Self::Sysconf),
            "crc32" => Some(Self::Crc32),
            "arc4random" => Some(Self::Arc4Random),
            "arc4random_buf" => Some(Self::Arc4RandomBuf),
            "rand" | "random" => Some(Self::Rand),
            "srand" | "srandom" => Some(Self::Srand),
            "getpid" => Some(Self::Getpid),
            "gettid" => Some(Self::Gettid),
            "getuid" | "geteuid" | "getgid" | "getegid" => Some(Self::Getuid),
            "time" => Some(Self::Time),
            "clock_gettime" => Some(Self::ClockGettime),
            "gettimeofday" => Some(Self::Gettimeofday),
            "localtime_r" | "gmtime_r" => Some(Self::LocaltimeR),
            "sleep" => Some(Self::Sleep),
            "usleep" => Some(Self::Usleep),
            "syscall" => Some(Self::Syscall),
            "__cxa_atexit" | "__cxa_finalize" | "atexit" => Some(Self::CxaAtexit),
            "__stack_chk_fail" => Some(Self::StackChkFail),
            "abort" => Some(Self::Abort),
            "exit" | "_exit" | "_Exit" => Some(Self::Exit),
            "memalign" | "aligned_alloc" => Some(Self::Memalign),
            "printf" => Some(Self::Printf),
            "fprintf" => Some(Self::Fprintf),
            "isalpha" => Some(Self::Ctype("isalpha")),
            "isdigit" => Some(Self::Ctype("isdigit")),
            "isalnum" => Some(Self::Ctype("isalnum")),
            "isspace" => Some(Self::Ctype("isspace")),
            "isupper" => Some(Self::Ctype("isupper")),
            "islower" => Some(Self::Ctype("islower")),
            "isxdigit" => Some(Self::Ctype("isxdigit")),
            "isprint" => Some(Self::Ctype("isprint")),
            "toupper" => Some(Self::Ctype("toupper")),
            "tolower" => Some(Self::Ctype("tolower")),
            "sin" => Some(Self::MathUnaryF64("sin")),
            "cos" => Some(Self::MathUnaryF64("cos")),
            "tan" => Some(Self::MathUnaryF64("tan")),
            "asin" => Some(Self::MathUnaryF64("asin")),
            "acos" => Some(Self::MathUnaryF64("acos")),
            "atan" => Some(Self::MathUnaryF64("atan")),
            "sinh" => Some(Self::MathUnaryF64("sinh")),
            "cosh" => Some(Self::MathUnaryF64("cosh")),
            "tanh" => Some(Self::MathUnaryF64("tanh")),
            "exp" => Some(Self::MathUnaryF64("exp")),
            "exp2" => Some(Self::MathUnaryF64("exp2")),
            "log" => Some(Self::MathUnaryF64("log")),
            "log2" => Some(Self::MathUnaryF64("log2")),
            "log10" => Some(Self::MathUnaryF64("log10")),
            "sqrt" => Some(Self::MathUnaryF64("sqrt")),
            "floor" => Some(Self::MathUnaryF64("floor")),
            "ceil" => Some(Self::MathUnaryF64("ceil")),
            "round" => Some(Self::MathUnaryF64("round")),
            "fabs" => Some(Self::MathUnaryF64("fabs")),
            "pow" => Some(Self::MathBinaryF64("pow")),
            "fmod" => Some(Self::MathBinaryF64("fmod")),
            "atan2" => Some(Self::MathBinaryF64("atan2")),
            "hypot" => Some(Self::MathBinaryF64("hypot")),
            "pthread_self" => Some(Self::PthreadSelf),
            "pthread_once" => Some(Self::PthreadOnce),
            "pthread_key_create" => Some(Self::PthreadKeyCreate),
            "pthread_setspecific" => Some(Self::PthreadSetSpecific),
            "pthread_getspecific" => Some(Self::PthreadGetSpecific),
            "pthread_create" => Some(Self::PthreadCreate),
            "pthread_join" => Some(Self::PthreadJoin),
            "pthread_detach" => Some(Self::PthreadDetach),
            "pthread_exit" => Some(Self::PthreadExit),
            "setjmp" | "_setjmp" | "sigsetjmp" => Some(Self::Setjmp),
            "longjmp" | "_longjmp" | "siglongjmp" => Some(Self::Longjmp),
            "sigaction" => Some(Self::Sigaction),
            "signal" | "bsd_signal" => Some(Self::Signal),
            "sigprocmask" | "pthread_sigmask" => Some(Self::Sigprocmask),
            "sigemptyset" => Some(Self::Sigemptyset),
            "sigfillset" => Some(Self::Sigfillset),
            "sigaddset" => Some(Self::Sigaddset),
            "sigdelset" => Some(Self::Sigdelset),
            "sigismember" => Some(Self::Sigismember),
            "AAssetManager_fromJava" => Some(Self::AAssetManagerFromJava),
            "AAssetManager_open" => Some(Self::AAssetManagerOpen),
            "AAssetManager_openDir" => Some(Self::AAssetManagerOpenDir),
            "AAsset_read" => Some(Self::AAssetRead),
            "AAsset_seek" | "AAsset_seek64" => Some(Self::AAssetSeek),
            "AAsset_getLength" | "AAsset_getLength64" => Some(Self::AAssetGetLength),
            "AAsset_getRemainingLength" | "AAsset_getRemainingLength64" => {
                Some(Self::AAssetGetRemainingLength)
            }
            "AAsset_getBuffer" => Some(Self::AAssetGetBuffer),
            "AAsset_close" => Some(Self::AAssetClose),
            "AAssetDir_getNextFileName" => Some(Self::AAssetDirGetNextFileName),
            "AAssetDir_rewind" => Some(Self::AAssetDirRewind),
            "AAssetDir_close" => Some(Self::AAssetDirClose),
            "adler32" => Some(Self::Adler32),
            "compressBound" => Some(Self::CompressBound),
            "compress" => Some(Self::Compress),
            "compress2" => Some(Self::Compress2),
            "uncompress" | "uncompress2" => Some(Self::Uncompress),
            "deflateInit_" => Some(Self::DeflateInit),
            "deflateInit2_" => Some(Self::DeflateInit2),
            "deflate" => Some(Self::Deflate),
            "deflateEnd" => Some(Self::DeflateEnd),
            "deflateReset" => Some(Self::DeflateReset),
            "inflateInit_" => Some(Self::InflateInit),
            "inflateInit2_" => Some(Self::InflateInit2),
            "inflate" => Some(Self::Inflate),
            "inflateEnd" => Some(Self::InflateEnd),
            "inflateReset" => Some(Self::InflateReset),
            "socket" => Some(Self::Socket),
            "bind" => Some(Self::Bind),
            "listen" => Some(Self::Listen),
            "accept" | "accept4" => Some(Self::Accept),
            "connect" => Some(Self::Connect),
            "send" => Some(Self::Send),
            "sendto" => Some(Self::Sendto),
            "recv" => Some(Self::Recv),
            "recvfrom" => Some(Self::Recvfrom),
            "shutdown" => Some(Self::Shutdown),
            "setsockopt" => Some(Self::Setsockopt),
            "getsockopt" => Some(Self::Getsockopt),
            "getsockname" => Some(Self::Getsockname),
            "getpeername" => Some(Self::Getpeername),
            "poll" => Some(Self::Poll),
            "epoll_create" | "epoll_create1" => Some(Self::EpollCreate1),
            "epoll_ctl" => Some(Self::EpollCtl),
            "epoll_wait" | "epoll_pwait" => Some(Self::EpollWait),
            "eventfd" => Some(Self::Eventfd),
            "pipe" | "pipe2" => Some(Self::Pipe),
            "getaddrinfo" => Some(Self::Getaddrinfo),
            "freeaddrinfo" => Some(Self::Freeaddrinfo),
            "inet_pton" => Some(Self::InetPton),
            "inet_ntop" => Some(Self::InetNtop),
            "htons" | "ntohs" => Some(Self::Htons),
            "htonl" | "ntohl" => Some(Self::Htonl),
            "getifaddrs" => Some(Self::Getifaddrs),
            "freeifaddrs" => Some(Self::Freeifaddrs),
            "fopen" | "fopen64" => Some(Self::Fopen),
            "fdopen" => Some(Self::Fdopen),
            "freopen" | "freopen64" => Some(Self::Freopen),
            "fclose" => Some(Self::Fclose),
            "fread" | "__fread_chk" | "fread_unlocked" => Some(Self::Fread),
            "fwrite" | "__fwrite_chk" | "fwrite_unlocked" => Some(Self::Fwrite),
            "fgets" | "fgets_unlocked" => Some(Self::Fgets),
            "__fgets_chk" => Some(Self::FgetsChk),
            "fputs" | "fputs_unlocked" => Some(Self::Fputs),
            "fgetc" | "getc" | "fgetc_unlocked" | "getc_unlocked" => Some(Self::Fgetc),
            "fputc" | "putc" | "fputc_unlocked" | "putc_unlocked" => Some(Self::Fputc),
            "ungetc" => Some(Self::Ungetc),
            "fseek" | "fseeko" | "fseeko64" => Some(Self::Fseek),
            "ftell" | "ftello" | "ftello64" => Some(Self::Ftell),
            "rewind" => Some(Self::Rewind),
            "feof" | "feof_unlocked" => Some(Self::Feof),
            "ferror" | "ferror_unlocked" => Some(Self::Ferror),
            "clearerr" | "clearerr_unlocked" => Some(Self::Clearerr),
            "fflush" | "fflush_unlocked" | "setvbuf" | "setbuf" => Some(Self::Fflush),
            "fileno" => Some(Self::Fileno),
            "fscanf" | "__isoc99_fscanf" => Some(Self::Fscanf),
            "getline" => Some(Self::Getline),
            "getdelim" => Some(Self::Getdelim),
            "lrand48" => Some(Self::Lrand48),
            "mrand48" => Some(Self::Mrand48),
            "srand48" => Some(Self::Srand48),
            "drand48" => Some(Self::Drand48),
            "dlopen" | "android_dlopen_ext" => Some(Self::Dlopen),
            "dlsym" => Some(Self::Dlsym),
            "dlclose" => Some(Self::Dlclose),
            "dlerror" => Some(Self::Dlerror),
            "dladdr" => Some(Self::Dladdr),
            "dl_iterate_phdr" => Some(Self::DlIteratePhdr),
            "opendir" => Some(Self::Opendir),
            "readdir" | "readdir64" => Some(Self::Readdir),
            "closedir" => Some(Self::Closedir),
            "rewinddir" => Some(Self::Rewinddir),
            "__system_property_find" => Some(Self::SystemPropertyFind),
            "__system_property_read" => Some(Self::SystemPropertyRead),
            "__system_property_read_callback" => Some(Self::SystemPropertyReadCallback),
            "flock"
            | "fcntl"
            | "ioctl"
            | "madvise"
            | "newlocale"
            | "uselocale"
            | "freelocale"
            | "mbtowc"
            | "fchown"
            | "chown"
            | "chmod"
            | "fchmod"
            | "fsync"
            | "fdatasync"
            | "getenv"
            | "alarm"
            | "prctl"
            | "eglGetProcAddress"
            | "android_set_abort_message"
            | "openlog"
            | "closelog"
            | "syslog"
            | "pthread_attr_init"
            | "pthread_attr_destroy"
            | "pthread_attr_setdetachstate"
            | "pthread_attr_setstacksize"
            | "pthread_mutex_init"
            | "pthread_mutex_lock"
            | "pthread_mutex_trylock"
            | "pthread_mutex_unlock"
            | "pthread_mutex_destroy"
            | "pthread_mutexattr_init"
            | "pthread_mutexattr_settype"
            | "pthread_mutexattr_destroy"
            | "pthread_rwlock_init"
            | "pthread_rwlock_rdlock"
            | "pthread_rwlock_wrlock"
            | "pthread_rwlock_unlock"
            | "pthread_rwlock_destroy"
            | "pthread_cond_init"
            | "pthread_cond_wait"
            | "pthread_cond_timedwait"
            | "pthread_cond_signal"
            | "pthread_cond_broadcast"
            | "pthread_cond_destroy"
            | "pthread_condattr_init"
            | "pthread_condattr_setclock"
            | "pthread_condattr_getclock"
            | "pthread_condattr_destroy"
            | "pthread_key_delete" => Some(Self::NoopZero),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StdioStream {
    pub fd: i32,
    pub eof: bool,
    pub err: bool,
    pub ungetc_buf: Option<u8>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirStream {
    pub entries: Vec<(String, u8)>,
    pub cursor: usize,
    pub dirent_buf: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct PropInfoEntry {
    pub name: String,
    pub value: String,
    pub name_ptr: u64,
    pub value_ptr: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct AssetStream {
    pub data: Vec<u8>,
    pub pos: usize,
    pub guest_buf: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct AssetDirStream {
    pub files: Vec<String>,
    pub cursor: usize,
    pub name_buf: u64,
}

enum ZlibStreamState {
    Deflate {
        compress: flate2::Compress,
        level: u32,
        zlib_header: bool,
    },
    Inflate {
        decompress: flate2::Decompress,
        window_bits: i32,
        gzip_header_pending: bool,
    },
}

/// 供 `dlopen`/`dlsym`/`dladdr`/`dl_iterate_phdr` 使用的已加载模块镜像信息
#[derive(Debug, Clone)]
pub struct DlModuleEntry {
    pub name: String,
    pub base_addr: u64,
    pub image_size: u64,
    pub symbols: HashMap<String, u64>,
}

/// HLE 跳板表：将外部符号或热点函数映射到 `TRAMPOLINE_BASE + slot * 4`
pub struct TrampolineTable {
    slots: Vec<HleBuiltin>,
    name_to_addr: HashMap<String, u64>,
    pthread_keys: HashMap<u32, u64>,
    next_pthread_key: u32,
    next_pthread_tid: u64,
    pending_threads: HashMap<u64, (u64, u64)>,
    pthread_exit_stack: Vec<(u64, u64)>,
    sigactions: HashMap<i32, [u8; 32]>,
    sigmask: u64,
    mmap_backings: HashMap<u64, MmapFileBacking>,
    stdio_streams: HashMap<u64, StdioStream>,
    dir_streams: HashMap<u64, DirStream>,
    asset_streams: HashMap<u64, AssetStream>,
    asset_dirs: HashMap<u64, AssetDirStream>,
    zlib_streams: HashMap<u64, ZlibStreamState>,
    prop_infos: HashMap<u64, PropInfoEntry>,
    loaded_modules: Vec<DlModuleEntry>,
    last_dlerror: Option<String>,
    lcg48_state: u64,
    unimplemented_hits: HashMap<String, u64>,
    pub errno_addr: u64,
    initialized: bool,
    /// 最近一次 HLE 跳板产生的内存/分配追踪事件（最多 3 条，如 `realloc` = `malloc` + `memcpy` + `free`），供 `BinaryTracer` 零分配消费
    pending_hle_events: [Option<crate::tracer::HleOpEvent>; 3],
    pending_hle_len: usize,
    /// 是否已输出过 `rand()/srand()` 确定性桩告警（每个实例仅提示一次，避免刷屏）
    rand_stub_warned: bool,
}

impl Default for TrampolineTable {
    fn default() -> Self {
        Self::new()
    }
}

/// 协作式 `pthread_create` 跳板地址（执行 `start_routine(arg)` 并把返回值写入 `*retval_slot`）
/// 入口约定：X0 = arg, X1 = start_routine, X2 = retval_slot, X30 = caller LR
/// 偏移 +16 (`str x0, [x19]`) 同时作为 `pthread_exit(retval)` 的退栈恢复点
const PTHREAD_CREATE_STUB_ADDR: u64 = TRAMPOLINE_BASE + PAGE_SIZE * 16 - 0x140;
/// `pthread_once` 回调执行跳板地址（位于 64KB 跳板区末尾，不占用符号槽位）
/// 指令序列：
/// ```asm
/// STP X29, X30, [SP, #-16]!   ; 0xA9BF7BFD
/// BLR X1                      ; 0xD63F0020 (调用 init_routine)
/// MOV X0, #0                  ; 0xD2800000 (返回 0)
/// LDP X29, X30, [SP], #16     ; 0xA8C17BFD
/// RET                         ; 0xD65F03C0
/// ```
const PTHREAD_ONCE_STUB_ADDR: u64 = TRAMPOLINE_BASE + PAGE_SIZE * 16 - 0x100;
const PROP_CALLBACK_STUB_ADDR: u64 = TRAMPOLINE_BASE + PAGE_SIZE * 16 - 0xE0;
const DL_ITERATE_STUB_ADDR: u64 = TRAMPOLINE_BASE + PAGE_SIZE * 16 - 0xC0;

impl TrampolineTable {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            name_to_addr: HashMap::new(),
            pthread_keys: HashMap::new(),
            next_pthread_key: 1,
            next_pthread_tid: 0x1001,
            pending_threads: HashMap::new(),
            pthread_exit_stack: Vec::new(),
            sigactions: HashMap::new(),
            sigmask: 0,
            mmap_backings: HashMap::new(),
            stdio_streams: HashMap::new(),
            dir_streams: HashMap::new(),
            asset_streams: HashMap::new(),
            asset_dirs: HashMap::new(),
            zlib_streams: HashMap::new(),
            prop_infos: HashMap::new(),
            loaded_modules: Vec::new(),
            last_dlerror: None,
            lcg48_state: 0x0000_1234_ABCD_330E,
            unimplemented_hits: HashMap::new(),
            errno_addr: crate::cpu::TLS_BASE + 0x100,
            initialized: false,
            pending_hle_events: [None, None, None],
            pending_hle_len: 0,
            rand_stub_warned: false,
        }
    }

    #[inline]
    pub(crate) fn push_hle_event(&mut self, ev: crate::tracer::HleOpEvent) {
        if self.pending_hle_len < self.pending_hle_events.len() {
            self.pending_hle_events[self.pending_hle_len] = Some(ev);
            self.pending_hle_len += 1;
        }
    }

    #[inline]
    pub(crate) fn drain_hle_events(&mut self) -> ([Option<crate::tracer::HleOpEvent>; 3], usize) {
        let out = self.pending_hle_events;
        let len = self.pending_hle_len;
        self.pending_hle_events = [None, None, None];
        self.pending_hle_len = 0;
        (out, len)
    }

    /// 同步已加载 ELF 模块元数据（供 `dlopen`/`dlsym`/`dladdr`/`dl_iterate_phdr` 查询）
    pub fn register_loaded_module(&mut self, entry: DlModuleEntry) {
        if let Some(existing) = self
            .loaded_modules
            .iter_mut()
            .find(|m| m.base_addr == entry.base_addr || m.name == entry.name)
        {
            *existing = entry;
        } else {
            self.loaded_modules.push(entry);
        }
    }

    /// 推进标准 POSIX 48-bit LCG (`X_{n+1} = (0x5DEECE66D * X_n + 0xB) mod 2^48`)
    fn next_lcg48(&mut self) -> u64 {
        self.lcg48_state = self
            .lcg48_state
            .wrapping_mul(0x5DEE_CE66D)
            .wrapping_add(0xB)
            & 0xFFFF_FFFF_FFFF;
        self.lcg48_state
    }

    /// 按符号名获取或预注册导入跳板地址（支持在 `emu.load` 之前或之后挂符号级 Hook）
    pub fn resolve_or_register_import<B: CpuBackend + ?Sized>(
        &mut self,
        uc: &mut B,
        mem: &mut MemoryManager,
        name: &str,
    ) -> EmuResult<u64> {
        if let Some(&addr) = self.name_to_addr.get(name) {
            return Ok(addr);
        }
        let action = HleBuiltin::from_symbol(name)
            .unwrap_or_else(|| HleBuiltin::UnimplementedStub(name.to_string()));
        self.register(uc, mem, name, action)
    }

    /// 映射跳板区内存（初始映射 64KB，按需使用）
    pub fn init<B: CpuBackend + ?Sized>(
        &mut self,
        uc: &mut B,
        mem: &mut MemoryManager,
    ) -> EmuResult<()> {
        if self.initialized {
            return Ok(());
        }
        mem.map_labeled(
            uc,
            TRAMPOLINE_BASE,
            PAGE_SIZE * 16,
            Prot::READ | Prot::EXEC,
            "[emu:hle_trampoline]",
            0,
        )?;
        let pthread_create_insns: [u32; 9] = [
            0xA9BE_7BFD, // stp x29, x30, [sp, #-32]!
            0xF900_0BF3, // str x19, [sp, #16]
            0xAA02_03F3, // mov x19, x2 (retval_slot)
            0xD63F_0020, // blr x1 (start_routine(x0=arg))
            0xF900_0260, // str x0, [x19]  <-- pthread_exit resume point (+16)
            0xD280_0000, // mov x0, #0
            0xF940_0BF3, // ldr x19, [sp, #16]
            0xA8C2_7BFD, // ldp x29, x30, [sp], #32
            ARM64_RET_INSN,
        ];
        for (i, insn) in pthread_create_insns.iter().enumerate() {
            mem.write_u32(uc, PTHREAD_CREATE_STUB_ADDR + (i as u64) * 4, *insn)?;
        }
        let stub_insns: [u32; 5] = [
            0xA9BF_7BFD, // stp x29, x30, [sp, #-16]!
            0xD63F_0020, // blr x1
            0xD280_0000, // mov x0, #0
            0xA8C1_7BFD, // ldp x29, x30, [sp], #16
            ARM64_RET_INSN,
        ];
        for (i, insn) in stub_insns.iter().enumerate() {
            mem.write_u32(uc, PTHREAD_ONCE_STUB_ADDR + (i as u64) * 4, *insn)?;
        }
        // __system_property_read_callback 跳板：调用 x9(x0=cookie, x1=name, x2=value, x3=serial)
        let prop_cb_insns: [u32; 4] = [
            0xA9BF_7BFD, // stp x29, x30, [sp, #-16]!
            0xD63F_0120, // blr x9
            0xA8C1_7BFD, // ldp x29, x30, [sp], #16
            ARM64_RET_INSN,
        ];
        for (i, insn) in prop_cb_insns.iter().enumerate() {
            mem.write_u32(uc, PROP_CALLBACK_STUB_ADDR + (i as u64) * 4, *insn)?;
        }
        // dl_iterate_phdr 回调循环跳板：x0=cb, x1=data, x2=info_array, x3=count (stride=64)
        let dl_iter_insns: [u32; 21] = [
            0xA9BD_7BFD, // [0]  stp x29, x30, [sp, #-48]!
            0xA901_53F3, // [1]  stp x19, x20, [sp, #16]
            0xA902_5BF5, // [2]  stp x21, x22, [sp, #32]
            0xAA00_03F3, // [3]  mov x19, x0 (cb)
            0xAA01_03F4, // [4]  mov x20, x1 (data)
            0xAA02_03F5, // [5]  mov x21, x2 (info ptr)
            0xAA03_03F6, // [6]  mov x22, x3 (count)
            0xD280_0000, // [7]  mov x0, #0
            0xB400_0136, // [8]  cbz x22, [17] (done)
            0xAA15_03E0, // [9]  mov x0, x21
            0xD280_0801, // [10] mov x1, #64
            0xAA14_03E2, // [11] mov x2, x20
            0xD63F_0260, // [12] blr x19
            0xB500_0080, // [13] cbnz x0, [17] (done)
            0x9101_02B5, // [14] add x21, x21, #64
            0xD100_06D6, // [15] sub x22, x22, #1
            0x17FF_FFF8, // [16] b [8] (loop)
            0xA942_5BF5, // [17] ldp x21, x22, [sp, #32]
            0xA941_53F3, // [18] ldp x19, x20, [sp, #16]
            0xA8C3_7BFD, // [19] ldp x29, x30, [sp], #48
            ARM64_RET_INSN,
        ];
        for (i, insn) in dl_iter_insns.iter().enumerate() {
            mem.write_u32(uc, DL_ITERATE_STUB_ADDR + (i as u64) * 4, *insn)?;
        }
        self.initialized = true;
        Ok(())
    }

    /// 注册一个符号到跳板槽位，并写入 `RET`（Unicorn）或 `SVC #0x1001`（非 Unicorn）
    pub fn register<B: CpuBackend + ?Sized>(
        &mut self,
        uc: &mut B,
        mem: &mut MemoryManager,
        name: &str,
        action: HleBuiltin,
    ) -> EmuResult<u64> {
        if !self.initialized {
            self.init(uc, mem)?;
        }
        if let Some(&addr) = self.name_to_addr.get(name) {
            return Ok(addr);
        }

        let slot_idx = self.slots.len() as u64;
        let addr = TRAMPOLINE_BASE + slot_idx * 4;
        let slot_insn = if uc.uses_svc_trampolines() {
            ARM64_SVC_HLE_INSN
        } else {
            ARM64_RET_INSN
        };
        mem.write_u32(uc, addr, slot_insn)?;
        self.slots.push(action);
        self.name_to_addr.insert(name.to_string(), addr);
        Ok(addr)
    }

    /// O(1) 按 PC 地址查找跳板动作
    #[inline]
    pub fn lookup(&self, pc: u64) -> Option<HleBuiltin> {
        if pc < TRAMPOLINE_BASE || !(pc - TRAMPOLINE_BASE).is_multiple_of(4) {
            return None;
        }
        let idx = ((pc - TRAMPOLINE_BASE) / 4) as usize;
        self.slots.get(idx).cloned()
    }

    pub fn get_addr(&self, name: &str) -> Option<u64> {
        self.name_to_addr.get(name).copied()
    }

    pub(crate) fn export_tape_state(&self) -> TrampolineTapeState {
        let mut slot_names = vec![String::new(); self.slots.len()];
        for (name, &addr) in &self.name_to_addr {
            if addr >= TRAMPOLINE_BASE && (addr - TRAMPOLINE_BASE).is_multiple_of(4) {
                let idx = ((addr - TRAMPOLINE_BASE) / 4) as usize;
                if idx < slot_names.len() {
                    slot_names[idx] = name.clone();
                }
            }
        }
        TrampolineTapeState {
            slot_names,
            pthread_keys: self.pthread_keys.iter().map(|(&k, &v)| (k, v)).collect(),
            next_pthread_key: self.next_pthread_key,
            next_pthread_tid: self.next_pthread_tid,
            pending_threads: self
                .pending_threads
                .iter()
                .map(|(&tid, &(f, a))| (tid, f, a))
                .collect(),
            pthread_exit_stack: self.pthread_exit_stack.clone(),
            sigactions: self.sigactions.iter().map(|(&k, &v)| (k, v)).collect(),
            sigmask: self.sigmask,
            mmap_backings: self
                .mmap_backings
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            stdio_streams: self
                .stdio_streams
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            dir_streams: self
                .dir_streams
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            asset_streams: self
                .asset_streams
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            asset_dirs: self
                .asset_dirs
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            prop_infos: self
                .prop_infos
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            last_dlerror: self.last_dlerror.clone(),
            loaded_modules: self.loaded_modules.clone(),
            lcg48_state: self.lcg48_state,
            errno_addr: self.errno_addr,
        }
    }

    /// 从 `.rtape` 恢复 HLE 跳板表元数据，并按目标后端重写全部 HLE 槽位指令（`RET` 或 `SVC #0x1001`）
    pub(crate) fn restore_tape_state_and_rewrite<B: CpuBackend + ?Sized>(
        &mut self,
        uc: &mut B,
        mem: &MemoryManager,
        state: TrampolineTapeState,
    ) -> EmuResult<()> {
        if (state.slot_names.len() as u64).saturating_mul(4) > TRAMPOLINE_SIZE {
            return Err(crate::error::EmuError::Backend(format!(
                "磁带 HLE 跳板槽位数量超界: {}",
                state.slot_names.len()
            )));
        }
        let slot_insn = if uc.uses_svc_trampolines() {
            ARM64_SVC_HLE_INSN
        } else {
            ARM64_RET_INSN
        };
        self.slots.clear();
        self.name_to_addr.clear();
        for (idx, name) in state.slot_names.into_iter().enumerate() {
            let addr = TRAMPOLINE_BASE + (idx as u64) * 4;
            let action = HleBuiltin::from_symbol(&name)
                .unwrap_or_else(|| HleBuiltin::UnimplementedStub(name.clone()));
            self.slots.push(action);
            if !name.is_empty() {
                self.name_to_addr.insert(name, addr);
            }
            mem.write_u32(uc, addr, slot_insn)?;
        }
        self.pthread_keys = state.pthread_keys.into_iter().collect();
        self.next_pthread_key = state.next_pthread_key;
        self.next_pthread_tid = state.next_pthread_tid;
        self.pending_threads = state
            .pending_threads
            .into_iter()
            .map(|(tid, f, a)| (tid, (f, a)))
            .collect();
        self.pthread_exit_stack = state.pthread_exit_stack;
        self.sigactions = state.sigactions.into_iter().collect();
        self.sigmask = state.sigmask;
        self.mmap_backings = state.mmap_backings.into_iter().collect();
        self.stdio_streams = state.stdio_streams.into_iter().collect();
        self.dir_streams = state.dir_streams.into_iter().collect();
        self.asset_streams = state.asset_streams.into_iter().collect();
        self.asset_dirs = state.asset_dirs.into_iter().collect();
        self.prop_infos = state.prop_infos.into_iter().collect();
        self.zlib_streams.clear();
        self.unimplemented_hits.clear();
        self.last_dlerror = state.last_dlerror;
        self.loaded_modules = state.loaded_modules;
        self.lcg48_state = state.lcg48_state;
        self.errno_addr = state.errno_addr;
        self.initialized = true;
        Ok(())
    }
}

/// HLE 跳板表可序列化状态（供 `.rtape` 磁带持久化与跨后端重写）
#[derive(Debug, Clone)]
pub(crate) struct TrampolineTapeState {
    pub slot_names: Vec<String>,
    pub pthread_keys: Vec<(u32, u64)>,
    pub next_pthread_key: u32,
    pub next_pthread_tid: u64,
    pub pending_threads: Vec<(u64, u64, u64)>,
    pub pthread_exit_stack: Vec<(u64, u64)>,
    pub sigactions: Vec<(i32, [u8; 32])>,
    pub sigmask: u64,
    pub mmap_backings: Vec<(u64, MmapFileBacking)>,
    pub stdio_streams: Vec<(u64, StdioStream)>,
    pub dir_streams: Vec<(u64, DirStream)>,
    pub asset_streams: Vec<(u64, AssetStream)>,
    pub asset_dirs: Vec<(u64, AssetDirStream)>,
    pub prop_infos: Vec<(u64, PropInfoEntry)>,
    pub last_dlerror: Option<String>,
    pub loaded_modules: Vec<DlModuleEntry>,
    pub lcg48_state: u64,
    pub errno_addr: u64,
}

/// 解析 C 风格十进制有符号整数前缀（兼容 `atoi` / `atol` / `atoll`）
fn parse_c_i64(s: &str) -> i64 {
    let bytes = s.trim_start().as_bytes();
    if bytes.is_empty() {
        return 0;
    }
    let (neg, start) = match bytes[0] {
        b'-' => (true, 1),
        b'+' => (false, 1),
        _ => (false, 0),
    };
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == start {
        return 0;
    }
    let digits = &s.trim_start()[start..end];
    let val = digits.parse::<i64>().unwrap_or(0);
    if neg {
        val.wrapping_neg()
    } else {
        val
    }
}

/// 精简版 AAPCS64 `sscanf(input, fmt, ...)`（支持 `%d/%i/%u/%ld/%lu/%lld/%llu/%x/%X/%s/%c`）
fn run_guest_sscanf<B: CpuBackend + ?Sized>(uc: &mut B, input: &str, fmt: &str) -> i32 {
    run_guest_scanf_core(uc, input, fmt).0
}

fn run_guest_scanf_core<B: CpuBackend + ?Sized>(
    uc: &mut B,
    input: &str,
    fmt: &str,
) -> (i32, usize) {
    let mut in_pos = 0usize;
    let in_bytes = input.as_bytes();
    let fmt_bytes = fmt.as_bytes();
    let mut fi = 0usize;
    let mut arg_idx = 2usize; // x2..x7, 然后栈参
    let mut matched = 0i32;

    let read_arg_ptr = |uc: &B, idx: usize| -> u64 {
        const REGS: [RegisterARM64; 8] = [
            RegisterARM64::X0,
            RegisterARM64::X1,
            RegisterARM64::X2,
            RegisterARM64::X3,
            RegisterARM64::X4,
            RegisterARM64::X5,
            RegisterARM64::X6,
            RegisterARM64::X7,
        ];
        if idx < 8 {
            uc.reg_read(REGS[idx]).unwrap_or(0)
        } else {
            let sp = uc.reg_read(RegisterARM64::SP).unwrap_or(0);
            let mut b = [0u8; 8];
            let _ = uc.mem_read(sp + ((idx - 8) as u64) * 8, &mut b);
            u64::from_le_bytes(b)
        }
    };

    while fi < fmt_bytes.len() {
        if fmt_bytes[fi].is_ascii_whitespace() {
            while fi < fmt_bytes.len() && fmt_bytes[fi].is_ascii_whitespace() {
                fi += 1;
            }
            while in_pos < in_bytes.len() && in_bytes[in_pos].is_ascii_whitespace() {
                in_pos += 1;
            }
            continue;
        }
        if fmt_bytes[fi] != b'%' {
            if in_pos < in_bytes.len() && in_bytes[in_pos] == fmt_bytes[fi] {
                in_pos += 1;
                fi += 1;
                continue;
            }
            break;
        }
        fi += 1;
        if fi < fmt_bytes.len() && fmt_bytes[fi] == b'%' {
            if in_pos < in_bytes.len() && in_bytes[in_pos] == b'%' {
                in_pos += 1;
                fi += 1;
                continue;
            }
            break;
        }
        let suppress = if fi < fmt_bytes.len() && fmt_bytes[fi] == b'*' {
            fi += 1;
            true
        } else {
            false
        };
        let mut width: Option<usize> = None;
        while fi < fmt_bytes.len() && fmt_bytes[fi].is_ascii_digit() {
            width = Some(width.unwrap_or(0) * 10 + (fmt_bytes[fi] - b'0') as usize);
            fi += 1;
        }
        let mut is_long = false;
        while fi < fmt_bytes.len() && matches!(fmt_bytes[fi], b'l' | b'h' | b'z') {
            if fmt_bytes[fi] == b'l' || fmt_bytes[fi] == b'z' {
                is_long = true;
            }
            fi += 1;
        }
        if fi >= fmt_bytes.len() {
            break;
        }
        let spec = fmt_bytes[fi];
        fi += 1;

        match spec {
            b'd' | b'i' | b'u' | b'x' | b'X' => {
                while in_pos < in_bytes.len() && in_bytes[in_pos].is_ascii_whitespace() {
                    in_pos += 1;
                }
                let max_end = width
                    .map(|w| in_pos.saturating_add(w).min(in_bytes.len()))
                    .unwrap_or(in_bytes.len());
                let mut neg = false;
                if in_pos < max_end && matches!(in_bytes[in_pos], b'-' | b'+') {
                    neg = in_bytes[in_pos] == b'-';
                    in_pos += 1;
                }
                let mut radix = if matches!(spec, b'x' | b'X') { 16u32 } else { 10u32 };
                if matches!(spec, b'i' | b'x' | b'X')
                    && in_pos + 1 < max_end
                    && in_bytes[in_pos] == b'0'
                    && matches!(in_bytes[in_pos + 1], b'x' | b'X')
                {
                    radix = 16;
                    in_pos += 2;
                }
                let dstart = in_pos;
                while in_pos < max_end && (in_bytes[in_pos] as char).is_digit(radix) {
                    in_pos += 1;
                }
                if in_pos == dstart {
                    break;
                }
                let raw = &input[dstart..in_pos];
                let uval = u64::from_str_radix(raw, radix).unwrap_or(0);
                let val = if neg {
                    (uval as i64).wrapping_neg() as u64
                } else {
                    uval
                };
                if !suppress {
                    let dst = read_arg_ptr(uc, arg_idx);
                    arg_idx += 1;
                    if dst != 0 {
                        if is_long {
                            let _ = uc.mem_write(dst, &val.to_le_bytes());
                        } else {
                            let _ = uc.mem_write(dst, &(val as u32).to_le_bytes());
                        }
                    }
                    matched += 1;
                }
            }
            b's' => {
                while in_pos < in_bytes.len() && in_bytes[in_pos].is_ascii_whitespace() {
                    in_pos += 1;
                }
                let max_end = width
                    .map(|w| in_pos.saturating_add(w).min(in_bytes.len()))
                    .unwrap_or(in_bytes.len());
                let start = in_pos;
                while in_pos < max_end && !in_bytes[in_pos].is_ascii_whitespace() {
                    in_pos += 1;
                }
                if in_pos == start {
                    break;
                }
                if !suppress {
                    let dst = read_arg_ptr(uc, arg_idx);
                    arg_idx += 1;
                    if dst != 0 {
                        let mut out = in_bytes[start..in_pos].to_vec();
                        out.push(0);
                        let _ = uc.mem_write(dst, &out);
                    }
                    matched += 1;
                }
            }
            b'c' => {
                let count = width.unwrap_or(1);
                if in_pos + count > in_bytes.len() {
                    break;
                }
                if !suppress {
                    let dst = read_arg_ptr(uc, arg_idx);
                    arg_idx += 1;
                    if dst != 0 {
                        let _ = uc.mem_write(dst, &in_bytes[in_pos..in_pos + count]);
                    }
                    matched += 1;
                }
                in_pos += count;
            }
            b'[' => {
                // 解析 scanset `%[...]` 或 `%[^...]`（常用于解析 `/proc/self/maps`、`/proc/cpuinfo` 等）
                let negate = if fi < fmt_bytes.len() && fmt_bytes[fi] == b'^' {
                    fi += 1;
                    true
                } else {
                    false
                };
                let mut set = [false; 256];
                let mut first = true;
                while fi < fmt_bytes.len() {
                    let b = fmt_bytes[fi];
                    fi += 1;
                    if b == b']' && !first {
                        break;
                    }
                    first = false;
                    if fi + 1 < fmt_bytes.len() && fmt_bytes[fi] == b'-' && fmt_bytes[fi + 1] != b']'
                    {
                        let end_b = fmt_bytes[fi + 1];
                        fi += 2;
                        for c in b..=end_b {
                            set[c as usize] = true;
                        }
                    } else {
                        set[b as usize] = true;
                    }
                }
                let max_end = width
                    .map(|w| in_pos.saturating_add(w).min(in_bytes.len()))
                    .unwrap_or(in_bytes.len());
                let start = in_pos;
                while in_pos < max_end {
                    let in_set = set[in_bytes[in_pos] as usize];
                    if (negate && in_set) || (!negate && !in_set) {
                        break;
                    }
                    in_pos += 1;
                }
                if in_pos == start {
                    break;
                }
                if !suppress {
                    let dst = read_arg_ptr(uc, arg_idx);
                    arg_idx += 1;
                    if dst != 0 {
                        let mut out = in_bytes[start..in_pos].to_vec();
                        out.push(0);
                        let _ = uc.mem_write(dst, &out);
                    }
                    matched += 1;
                }
            }
            b'f' | b'g' | b'e' => {
                while in_pos < in_bytes.len() && in_bytes[in_pos].is_ascii_whitespace() {
                    in_pos += 1;
                }
                let max_end = width
                    .map(|w| in_pos.saturating_add(w).min(in_bytes.len()))
                    .unwrap_or(in_bytes.len());
                let start = in_pos;
                if in_pos < max_end && matches!(in_bytes[in_pos], b'+' | b'-') {
                    in_pos += 1;
                }
                while in_pos < max_end
                    && (in_bytes[in_pos].is_ascii_digit()
                        || matches!(in_bytes[in_pos], b'.' | b'e' | b'E' | b'+' | b'-'))
                {
                    in_pos += 1;
                }
                if in_pos == start {
                    break;
                }
                let val: f64 = input[start..in_pos].parse().unwrap_or(0.0);
                if !suppress {
                    let dst = read_arg_ptr(uc, arg_idx);
                    arg_idx += 1;
                    if dst != 0 {
                        if is_long {
                            let _ = uc.mem_write(dst, &val.to_le_bytes());
                        } else {
                            let _ = uc.mem_write(dst, &(val as f32).to_le_bytes());
                        }
                    }
                    matched += 1;
                }
            }
            b'n' => {
                if !suppress {
                    let dst = read_arg_ptr(uc, arg_idx);
                    arg_idx += 1;
                    if dst != 0 {
                        let _ = uc.mem_write(dst, &(in_pos as i32).to_le_bytes());
                    }
                }
            }
            _ => break,
        }
    }
    (matched, in_pos)
}

/// 按 AAPCS64 规则解析 `snprintf`/`sprintf` 变参格式化字符串
/// 支持 `%s`、`%d`/`%i`/`%u`、`%x`/`%X`、`%p`、`%c`、`%%` 以及宽度/零填充（如 `%02x`、`%08X`）
fn format_guest_printf<B: CpuBackend + ?Sized>(
    uc: &B,
    fmt: &str,
    first_gpr_idx: usize,
) -> String {
    const GPRS: [RegisterARM64; 8] = [
        RegisterARM64::X0,
        RegisterARM64::X1,
        RegisterARM64::X2,
        RegisterARM64::X3,
        RegisterARM64::X4,
        RegisterARM64::X5,
        RegisterARM64::X6,
        RegisterARM64::X7,
    ];
    let sp = uc.reg_read(RegisterARM64::SP).unwrap_or(0);
    let mut gpr = first_gpr_idx;
    let mut stack_off = 0u64;

    format_guest_printf_with_reader(uc, fmt, || {
        if gpr < GPRS.len() {
            let v = uc.reg_read(GPRS[gpr]).unwrap_or(0);
            gpr += 1;
            v
        } else {
            let mut buf = [0u8; 8];
            let v = if uc.mem_read(sp + stack_off, &mut buf).is_ok() {
                u64::from_le_bytes(buf)
            } else {
                0
            };
            stack_off += 8;
            v
        }
    })
}

/// 按 AAPCS64 `va_list` 结构解析 `vsnprintf`/`__vsnprintf_chk`/`vasprintf`
/// AAPCS64 `va_list` (32B): `{ __stack: u64, __gr_top: u64, __vr_top: u64, __gr_offs: i32, __vr_offs: i32 }`
fn format_guest_vprintf<B: CpuBackend + ?Sized>(uc: &B, fmt: &str, va_list_ptr: u64) -> String {
    if va_list_ptr == 0 {
        return fmt.to_string();
    }
    let mut raw = [0u8; 32];
    if uc.mem_read(va_list_ptr, &mut raw).is_err() {
        return fmt.to_string();
    }
    let mut stack_ptr = u64::from_le_bytes(raw[0..8].try_into().unwrap());
    let gr_top = u64::from_le_bytes(raw[8..16].try_into().unwrap());
    let mut gr_offs = i32::from_le_bytes(raw[24..28].try_into().unwrap());

    format_guest_printf_with_reader(uc, fmt, || {
        let addr = if gr_offs < 0 {
            let a = (gr_top as i64).wrapping_add(gr_offs as i64) as u64;
            gr_offs = gr_offs.saturating_add(8);
            a
        } else {
            let a = stack_ptr;
            stack_ptr = stack_ptr.wrapping_add(8);
            a
        };
        let mut buf = [0u8; 8];
        if uc.mem_read(addr, &mut buf).is_ok() {
            u64::from_le_bytes(buf)
        } else {
            0
        }
    })
}

fn format_guest_printf_with_reader<B: CpuBackend + ?Sized, F>(
    uc: &B,
    fmt: &str,
    mut next_arg: F,
) -> String
where
    F: FnMut() -> u64,
{

    let bytes = fmt.as_bytes();
    let mut out = String::with_capacity(fmt.len() + 32);
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] != b'%' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        i += 1;
        if i >= bytes.len() {
            break;
        }
        if bytes[i] == b'%' {
            out.push('%');
            i += 1;
            continue;
        }

        let mut zero_pad = false;
        let mut width: usize = 0;
        if bytes[i] == b'0' {
            zero_pad = true;
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            width = width * 10 + (bytes[i] - b'0') as usize;
            i += 1;
        }
        // 跳过精度段（如 %.4s）
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        let mut is_long = false;
        while i < bytes.len() && matches!(bytes[i], b'l' | b'h' | b'z') {
            if bytes[i] == b'l' || bytes[i] == b'z' {
                is_long = true;
            }
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        let spec = bytes[i] as char;
        i += 1;
        let val = next_arg();
        let formatted = match spec {
            's' => {
                if val == 0 {
                    "(null)".to_string()
                } else {
                    read_guest_cstr(uc, val, 4096)
                }
            }
            'd' | 'i' => {
                if is_long {
                    format!("{}", val as i64)
                } else {
                    format!("{}", val as i32)
                }
            }
            'u' => {
                if is_long {
                    format!("{val}")
                } else {
                    format!("{}", val as u32)
                }
            }
            'x' => {
                if is_long {
                    format!("{val:x}")
                } else {
                    format!("{:x}", val as u32)
                }
            }
            'X' => {
                if is_long {
                    format!("{val:X}")
                } else {
                    format!("{:X}", val as u32)
                }
            }
            'p' => format!("0x{val:x}"),
            'c' => format!("{}", (val as u8) as char),
            _ => format!("%{spec}"),
        };

        if width > formatted.len() {
            let pad_ch = if zero_pad { '0' } else { ' ' };
            for _ in 0..(width - formatted.len()) {
                out.push(pad_ch);
            }
        }
        out.push_str(&formatted);
    }

    out
}

/// 按槽位地址分发 HLE 跳板（供 Unicorn Code Hook 与非 Unicorn `SVC #0x1001` 外部陷阱循环共用）
pub fn dispatch_trampoline_slot<B, D, F>(uc: &mut B, slot_addr: u64, get_state: &mut F) -> bool
where
    B: BackendWithData<D> + ?Sized,
    F: FnMut(&mut D) -> (&mut TrampolineTable, &mut MemoryManager, &mut VirtualFs),
{
    let action = {
        let (table, _, _) = get_state(uc.get_data_mut());
        table.lookup(slot_addr)
    };
    let Some(builtin) = action else {
        return false;
    };
    execute_hle_builtin(uc, builtin, get_state);
    true
}

/// 挂载 HLE 跳板区 Code Hook
///
/// 当 Guest 执行 `BLR` / `BR` 跳转到 `TRAMPOLINE_BASE..TRAMPOLINE_BASE + TRAMPOLINE_SIZE` 时，
/// 在执行该槽位的 `RET` 指令前拦截并运行对应的原生 Rust 实现，将结果写入 `X0`。
pub fn attach_trampoline_hook<'a, D, F>(
    uc: &mut Unicorn<'a, D>,
    mut get_state: F,
) -> EmuResult<UcHookId>
where
    F: FnMut(&mut D) -> (&mut TrampolineTable, &mut MemoryManager, &mut VirtualFs) + 'a,
{
    let hook_id = uc.add_code_hook(
        TRAMPOLINE_BASE,
        TRAMPOLINE_BASE + TRAMPOLINE_SIZE,
        move |uc, addr, _size| {
            dispatch_trampoline_slot(uc, addr, &mut get_state);
        },
    )?;

    Ok(hook_id)
}

/// 将标准 Linux 负错误码转换为 C 库约定（设置 `*__errno() = -res` 并返回 `-1`）
#[inline]
fn c_errno_ret<B: CpuBackend + ?Sized>(uc: &mut B, errno_addr: u64, res: i64) -> u64 {
    if res < 0 {
        let err = (-res) as i32;
        let _ = uc.mem_write(errno_addr, &err.to_le_bytes());
        u64::MAX
    } else {
        res as u64
    }
}

/// 按 Linux AArch64 ABI 填充 128 字节 `struct stat`
#[inline]
fn write_aarch64_stat<B: CpuBackend + ?Sized>(
    uc: &mut B,
    stat_buf: u64,
    mode: u32,
    size: u64,
) -> bool {
    let mut st = [0u8; 128];
    st[16..20].copy_from_slice(&mode.to_le_bytes());
    st[48..56].copy_from_slice(&size.to_le_bytes());
    st[56..60].copy_from_slice(&4096i32.to_le_bytes()); // st_blksize
    st[64..72].copy_from_slice(&(size.div_ceil(512) as i64).to_le_bytes()); // st_blocks
    uc.mem_write(stat_buf, &st).is_ok()
}

/// 标准 IEEE 802.3 zlib `crc32` 计算（多项式 `0xEDB88320`）
fn zlib_crc32(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            let mask = (c & 1).wrapping_neg();
            c = (c >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !c
}

/// 将 Linux `PROT_READ/WRITE/EXEC` 转换为 Unicorn `Prot`
#[inline]
fn linux_prot_to_uc(prot: u64) -> Prot {
    let mut p = Prot::NONE;
    if prot & 0x1 != 0 {
        p |= Prot::READ;
    }
    if prot & 0x2 != 0 {
        p |= Prot::WRITE;
    }
    if prot & 0x4 != 0 {
        p |= Prot::EXEC;
    }
    if p == Prot::NONE {
        Prot::READ | Prot::WRITE
    } else {
        p
    }
}

/// 执行具体的原生 Rust HLE 逻辑，并按 AAPCS64 将返回值写入 X0
fn execute_hle_builtin<B, D, F>(uc: &mut B, builtin: HleBuiltin, get_state: &mut F)
where
    B: BackendWithData<D> + ?Sized,
    F: FnMut(&mut D) -> (&mut TrampolineTable, &mut MemoryManager, &mut VirtualFs),
{
    let x0 = uc.reg_read(RegisterARM64::X0).unwrap_or(0);
    let x1 = uc.reg_read(RegisterARM64::X1).unwrap_or(0);
    let x2 = uc.reg_read(RegisterARM64::X2).unwrap_or(0);
    let x3 = uc.reg_read(RegisterARM64::X3).unwrap_or(0);
    let x4 = uc.reg_read(RegisterARM64::X4).unwrap_or(0);
    let x5 = uc.reg_read(RegisterARM64::X5).unwrap_or(0);
    let lr = uc.reg_read(RegisterARM64::LR).unwrap_or(0);
    let caller_pc = if lr == crate::cpu::RET_MAGIC_ADDR || lr < 4 {
        0
    } else {
        lr - 4
    };
    // 防御性清空上次可能未消费的 HLE 事件（避免独立调用 dispatch_trampoline_slot 时累积）
    {
        let (tramp, _, _) = get_state(uc.get_data_mut());
        let _ = tramp.drain_hle_events();
    }

    let ret: u64 = match builtin {
        HleBuiltin::Malloc => {
            let alloc_sz = x0.max(1);
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap(16 + alloc_sz).unwrap_or(0)
            };
            if raw != 0 {
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &alloc_sz.to_le_bytes());
                let ptr = raw + 16;
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(ptr, alloc_sz, caller_pc, "malloc");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 0,
                    aux_subop: 0,
                    dst: ptr,
                    src: 0,
                    len: alloc_sz,
                    caller_pc,
                });
                ptr
            } else {
                0
            }
        }
        HleBuiltin::Calloc => {
            let total = x0.saturating_mul(x1).max(1);
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap(16 + total).unwrap_or(0)
            };
            if raw != 0 {
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &total.to_le_bytes());
                let zeros = vec![0u8; total as usize];
                let _ = uc.mem_write(raw + 16, &zeros);
                let ptr = raw + 16;
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(ptr, total, caller_pc, "calloc");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 1,
                    aux_subop: 0,
                    dst: ptr,
                    src: 0,
                    len: total,
                    caller_pc,
                });
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMemset,
                    subop_or_byte: 0,
                    aux_subop: 2,
                    dst: ptr,
                    src: 0,
                    len: total,
                    caller_pc,
                });
                ptr
            } else {
                0
            }
        }
        HleBuiltin::Realloc => {
            let old_ptr = x0;
            if x1 == 0 {
                if old_ptr != 0 {
                    let (tramp, mem, _) = get_state(uc.get_data_mut());
                    mem.hle_free(old_ptr);
                    mem.record_heap_free(old_ptr, caller_pc);
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleFree,
                        subop_or_byte: 0,
                        aux_subop: 0,
                        dst: old_ptr,
                        src: 0,
                        len: 0,
                        caller_pc,
                    });
                }
                0
            } else {
                let new_size = x1;
                let raw = {
                    let (_, mem, _) = get_state(uc.get_data_mut());
                    mem.alloc_heap(16 + new_size).unwrap_or(0)
                };
                if raw != 0 {
                    let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                    let _ = uc.mem_write(raw + 8, &new_size.to_le_bytes());
                    let new_ptr = raw + 16;
                    let mut copied_len = 0u64;
                    if old_ptr >= 16 {
                        let mut hdr = [0u8; 16];
                        if uc.mem_read(old_ptr - 16, &mut hdr).is_ok() {
                            let magic = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
                            let old_sz = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
                            if magic == 0x4845_4150_424C_4B31 {
                                let copy_len = old_sz.min(new_size) as usize;
                                let mut data = vec![0u8; copy_len];
                                if uc.mem_read(old_ptr, &mut data).is_ok()
                                    && uc.mem_write(new_ptr, &data).is_ok()
                                {
                                    copied_len = copy_len as u64;
                                }
                            }
                        }
                    }
                    let (tramp, mem, _) = get_state(uc.get_data_mut());
                    mem.record_heap_alloc(new_ptr, new_size, caller_pc, "realloc");
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleMalloc,
                        subop_or_byte: 2,
                        aux_subop: 0,
                        dst: new_ptr,
                        src: old_ptr,
                        len: new_size,
                        caller_pc,
                    });
                    if copied_len > 0 {
                        tramp.push_hle_event(crate::tracer::HleOpEvent {
                            kind: crate::tracer::ExtKind::HleMemcpy,
                            subop_or_byte: 2,
                            aux_subop: 0,
                            dst: new_ptr,
                            src: old_ptr,
                            len: copied_len,
                            caller_pc,
                        });
                    }
                    if old_ptr != 0 {
                        mem.hle_free(old_ptr);
                        mem.record_heap_free(old_ptr, caller_pc);
                        tramp.push_hle_event(crate::tracer::HleOpEvent {
                            kind: crate::tracer::ExtKind::HleFree,
                            subop_or_byte: 0,
                            aux_subop: 0,
                            dst: old_ptr,
                            src: 0,
                            len: 0,
                            caller_pc,
                        });
                    }
                    new_ptr
                } else {
                    0
                }
            }
        }
        HleBuiltin::PosixMemalign => {
            // int posix_memalign(void **memptr, size_t alignment, size_t size)
            let alignment = x1.max(16);
            let alloc_sz = x2.max(1);
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap_aligned_with_header(alloc_sz, alignment)
                    .unwrap_or(0)
            };
            if raw != 0 && x0 != 0 {
                let aligned = raw + 16;
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &alloc_sz.to_le_bytes());
                let _ = uc.mem_write(x0, &aligned.to_le_bytes());
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(aligned, alloc_sz, caller_pc, "posix_memalign");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 3,
                    aux_subop: 0,
                    dst: aligned,
                    src: 0,
                    len: alloc_sz,
                    caller_pc,
                });
                0
            } else {
                if raw != 0 {
                    let (_, mem, _) = get_state(uc.get_data_mut());
                    mem.free_heap(raw);
                }
                12 // ENOMEM
            }
        }
        HleBuiltin::Free => {
            if x0 != 0 {
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.hle_free(x0);
                mem.record_heap_free(x0, caller_pc);
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleFree,
                    subop_or_byte: 0,
                    aux_subop: 0,
                    dst: x0,
                    src: 0,
                    len: 0,
                    caller_pc,
                });
            }
            0
        }
        HleBuiltin::CxaAtexit | HleBuiltin::NoopZero => 0,
        HleBuiltin::Memcpy | HleBuiltin::Memmove => {
            let len = x2 as usize;
            if len > 0 {
                let mut buf = vec![0u8; len];
                if uc.mem_read(x1, &mut buf).is_ok() && uc.mem_write(x0, &buf).is_ok() {
                    let subop = if matches!(builtin, HleBuiltin::Memmove) {
                        1
                    } else {
                        0
                    };
                    let (tramp, _, _) = get_state(uc.get_data_mut());
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleMemcpy,
                        subop_or_byte: subop,
                        aux_subop: 0,
                        dst: x0,
                        src: x1,
                        len: x2,
                        caller_pc,
                    });
                }
            }
            x0
        }
        HleBuiltin::Memset => {
            let len = x2 as usize;
            if len > 0 {
                let byte = (x1 & 0xFF) as u8;
                let buf = vec![byte; len];
                if uc.mem_write(x0, &buf).is_ok() {
                    let (tramp, _, _) = get_state(uc.get_data_mut());
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleMemset,
                        subop_or_byte: byte,
                        aux_subop: 0,
                        dst: x0,
                        src: byte as u64,
                        len: x2,
                        caller_pc,
                    });
                }
            }
            x0
        }
        HleBuiltin::Bzero => {
            let len = x1 as usize;
            if len > 0 {
                let buf = vec![0u8; len];
                if uc.mem_write(x0, &buf).is_ok() {
                    let (tramp, _, _) = get_state(uc.get_data_mut());
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleMemset,
                        subop_or_byte: 0,
                        aux_subop: 1,
                        dst: x0,
                        src: 0,
                        len: x1,
                        caller_pc,
                    });
                }
            }
            0
        }
        HleBuiltin::Memcmp => {
            let len = x2 as usize;
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];
            if len > 0 && uc.mem_read(x0, &mut a).is_ok() && uc.mem_read(x1, &mut b).is_ok() {
                match a.cmp(&b) {
                    std::cmp::Ordering::Less => (-1i64) as u64,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                }
            } else {
                0
            }
        }
        HleBuiltin::Memchr => {
            let needle = (x1 & 0xFF) as u8;
            let len = (x2 as usize).min(4 * 1024 * 1024);
            let mut buf = vec![0u8; len];
            if len > 0 && uc.mem_read(x0, &mut buf).is_ok() {
                buf.iter()
                    .position(|&b| b == needle)
                    .map(|p| x0 + p as u64)
                    .unwrap_or(0)
            } else {
                0
            }
        }
        HleBuiltin::Strlen => read_guest_cstr_bytes(uc, x0, 1024 * 1024).len() as u64,
        HleBuiltin::Strnlen => {
            let max_len = (x1 as usize).min(1024 * 1024);
            read_guest_cstr_bytes(uc, x0, max_len).len() as u64
        }
        HleBuiltin::Strcmp => {
            let a = read_guest_cstr_bytes(uc, x0, 1024 * 1024);
            let b = read_guest_cstr_bytes(uc, x1, 1024 * 1024);
            match a.cmp(&b) {
                std::cmp::Ordering::Less => (-1i64) as u64,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }
        HleBuiltin::Strncmp => {
            let n = (x2 as usize).min(1024 * 1024);
            let a = read_guest_cstr_bytes(uc, x0, n);
            let b = read_guest_cstr_bytes(uc, x1, n);
            match a.cmp(&b) {
                std::cmp::Ordering::Less => (-1i64) as u64,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }
        HleBuiltin::Strcasecmp => {
            let a: Vec<u8> = read_guest_cstr_bytes(uc, x0, 1024 * 1024)
                .into_iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            let b: Vec<u8> = read_guest_cstr_bytes(uc, x1, 1024 * 1024)
                .into_iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            match a.cmp(&b) {
                std::cmp::Ordering::Less => (-1i64) as u64,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }
        HleBuiltin::Strncasecmp => {
            let n = (x2 as usize).min(1024 * 1024);
            let a: Vec<u8> = read_guest_cstr_bytes(uc, x0, n)
                .into_iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            let b: Vec<u8> = read_guest_cstr_bytes(uc, x1, n)
                .into_iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            match a.cmp(&b) {
                std::cmp::Ordering::Less => (-1i64) as u64,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }
        HleBuiltin::Strcpy => {
            let mut src = read_guest_cstr_bytes(uc, x1, 1024 * 1024);
            src.push(0);
            let _ = uc.mem_write(x0, &src);
            x0
        }
        HleBuiltin::Strncpy => {
            let n = (x2 as usize).min(1024 * 1024);
            let src = read_guest_cstr_bytes(uc, x1, n);
            let mut dst = vec![0u8; n];
            let copy_len = src.len().min(n);
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
            let _ = uc.mem_write(x0, &dst);
            x0
        }
        HleBuiltin::Strcat => {
            let dst_len = read_guest_cstr_bytes(uc, x0, 1024 * 1024).len();
            let mut src = read_guest_cstr_bytes(uc, x1, 1024 * 1024);
            src.push(0);
            let _ = uc.mem_write(x0 + dst_len as u64, &src);
            x0
        }
        HleBuiltin::Strchr => {
            let needle = (x1 & 0xFF) as u8;
            let bytes = read_guest_cstr_bytes(uc, x0, 1024 * 1024);
            if needle == 0 {
                x0 + bytes.len() as u64
            } else {
                bytes
                    .iter()
                    .position(|&b| b == needle)
                    .map(|p| x0 + p as u64)
                    .unwrap_or(0)
            }
        }
        HleBuiltin::Strrchr => {
            let needle = (x1 & 0xFF) as u8;
            let bytes = read_guest_cstr_bytes(uc, x0, 1024 * 1024);
            if needle == 0 {
                x0 + bytes.len() as u64
            } else {
                bytes
                    .iter()
                    .rposition(|&b| b == needle)
                    .map(|p| x0 + p as u64)
                    .unwrap_or(0)
            }
        }
        HleBuiltin::Strstr => {
            let hay = read_guest_cstr(uc, x0, 1024 * 1024);
            let needle = read_guest_cstr(uc, x1, 1024 * 1024);
            if needle.is_empty() {
                x0
            } else {
                hay.find(&needle).map(|p| x0 + p as u64).unwrap_or(0)
            }
        }
        HleBuiltin::Strspn => {
            let s = read_guest_cstr_bytes(uc, x0, 1024 * 1024);
            let accept = read_guest_cstr_bytes(uc, x1, 1024);
            s.iter().take_while(|b| accept.contains(b)).count() as u64
        }
        HleBuiltin::Strcspn => {
            let s = read_guest_cstr_bytes(uc, x0, 1024 * 1024);
            let reject = read_guest_cstr_bytes(uc, x1, 1024);
            s.iter().take_while(|b| !reject.contains(b)).count() as u64
        }
        HleBuiltin::Strdup => {
            let mut bytes = read_guest_cstr_bytes(uc, x0, 1024 * 1024);
            bytes.push(0);
            let alloc_sz = bytes.len() as u64;
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap(16 + alloc_sz).unwrap_or(0)
            };
            if raw != 0 {
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &alloc_sz.to_le_bytes());
                let _ = uc.mem_write(raw + 16, &bytes);
                let ptr = raw + 16;
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(ptr, alloc_sz, caller_pc, "strdup");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 0,
                    aux_subop: 0,
                    dst: ptr,
                    src: 0,
                    len: alloc_sz,
                    caller_pc,
                });
                ptr
            } else {
                0
            }
        }
        HleBuiltin::Snprintf => {
            // int snprintf(char *str, size_t size, const char *format, ...)
            let max_sz = x1 as usize;
            let fmt = read_guest_cstr(uc, x2, 4096);
            let formatted = format_guest_printf(uc, &fmt, 3);
            if max_sz > 0 && x0 != 0 {
                let bytes = formatted.as_bytes();
                let copy_len = bytes.len().min(max_sz - 1);
                let mut out = Vec::with_capacity(copy_len + 1);
                out.extend_from_slice(&bytes[..copy_len]);
                out.push(0);
                let _ = uc.mem_write(x0, &out);
            }
            formatted.len() as u64
        }
        HleBuiltin::SnprintfChk => {
            // int __snprintf_chk(char *dst, size_t size, int flags, size_t destlen, const char *fmt, ...)
            let max_sz = x1 as usize;
            let fmt_ptr = uc.reg_read(RegisterARM64::X4).unwrap_or(0);
            let fmt = read_guest_cstr(uc, fmt_ptr, 4096);
            let formatted = format_guest_printf(uc, &fmt, 5);
            if max_sz > 0 && x0 != 0 {
                let bytes = formatted.as_bytes();
                let copy_len = bytes.len().min(max_sz - 1);
                let mut out = Vec::with_capacity(copy_len + 1);
                out.extend_from_slice(&bytes[..copy_len]);
                out.push(0);
                let _ = uc.mem_write(x0, &out);
            }
            formatted.len() as u64
        }
        HleBuiltin::Sprintf => {
            // int sprintf(char *str, const char *format, ...)
            let fmt = read_guest_cstr(uc, x1, 4096);
            let formatted = format_guest_printf(uc, &fmt, 2);
            if x0 != 0 {
                let mut out = formatted.as_bytes().to_vec();
                out.push(0);
                let _ = uc.mem_write(x0, &out);
            }
            formatted.len() as u64
        }
        HleBuiltin::SprintfChk => {
            // int __sprintf_chk(char *dst, int flags, size_t destlen, const char *fmt, ...)
            let fmt_ptr = uc.reg_read(RegisterARM64::X3).unwrap_or(0);
            let fmt = read_guest_cstr(uc, fmt_ptr, 4096);
            let formatted = format_guest_printf(uc, &fmt, 4);
            if x0 != 0 {
                let mut out = formatted.as_bytes().to_vec();
                out.push(0);
                let _ = uc.mem_write(x0, &out);
            }
            formatted.len() as u64
        }
        HleBuiltin::Asprintf => {
            // int asprintf(char **strp, const char *format, ...)
            let fmt = read_guest_cstr(uc, x1, 4096);
            let formatted = format_guest_printf(uc, &fmt, 2);
            let mut out = formatted.as_bytes().to_vec();
            out.push(0);
            let alloc_sz = out.len() as u64;
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap(16 + alloc_sz).unwrap_or(0)
            };
            if raw != 0 {
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &alloc_sz.to_le_bytes());
                let ptr = raw + 16;
                let _ = uc.mem_write(ptr, &out);
                if x0 != 0 {
                    let _ = uc.mem_write(x0, &ptr.to_le_bytes());
                }
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(ptr, alloc_sz, caller_pc, "asprintf");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 0,
                    aux_subop: 0,
                    dst: ptr,
                    src: 0,
                    len: alloc_sz,
                    caller_pc,
                });
                formatted.len() as u64
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Vasprintf => {
            // int vasprintf(char **strp, const char *format, va_list ap)
            let fmt = read_guest_cstr(uc, x1, 4096);
            let formatted = format_guest_vprintf(uc, &fmt, x2);
            let mut out = formatted.as_bytes().to_vec();
            out.push(0);
            let alloc_sz = out.len() as u64;
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap(16 + alloc_sz).unwrap_or(0)
            };
            if raw != 0 {
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &alloc_sz.to_le_bytes());
                let ptr = raw + 16;
                let _ = uc.mem_write(ptr, &out);
                if x0 != 0 {
                    let _ = uc.mem_write(x0, &ptr.to_le_bytes());
                }
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(ptr, alloc_sz, caller_pc, "vasprintf");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 0,
                    aux_subop: 0,
                    dst: ptr,
                    src: 0,
                    len: alloc_sz,
                    caller_pc,
                });
                formatted.len() as u64
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Vsnprintf => {
            // int vsnprintf(char *str, size_t size, const char *fmt, va_list ap)
            let max_sz = x1 as usize;
            let fmt = read_guest_cstr(uc, x2, 4096);
            let ap_ptr = uc.reg_read(RegisterARM64::X3).unwrap_or(0);
            let formatted = format_guest_vprintf(uc, &fmt, ap_ptr);
            if max_sz > 0 && x0 != 0 {
                let bytes = formatted.as_bytes();
                let copy_len = bytes.len().min(max_sz - 1);
                let mut out = Vec::with_capacity(copy_len + 1);
                out.extend_from_slice(&bytes[..copy_len]);
                out.push(0);
                let _ = uc.mem_write(x0, &out);
            }
            formatted.len() as u64
        }
        HleBuiltin::VsnprintfChk => {
            // int __vsnprintf_chk(char *dst, size_t size, int flags, size_t destlen, const char *fmt, va_list ap)
            let max_sz = x1 as usize;
            let fmt_ptr = uc.reg_read(RegisterARM64::X4).unwrap_or(0);
            let ap_ptr = uc.reg_read(RegisterARM64::X5).unwrap_or(0);
            let fmt = read_guest_cstr(uc, fmt_ptr, 4096);
            let formatted = format_guest_vprintf(uc, &fmt, ap_ptr);
            if max_sz > 0 && x0 != 0 {
                let bytes = formatted.as_bytes();
                let copy_len = bytes.len().min(max_sz - 1);
                let mut out = Vec::with_capacity(copy_len + 1);
                out.extend_from_slice(&bytes[..copy_len]);
                out.push(0);
                let _ = uc.mem_write(x0, &out);
            }
            formatted.len() as u64
        }
        HleBuiltin::Strerror => {
            let msg_addr = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table.errno_addr + 8
            };
            let msg = format!("errno {}\0", x0 as i32);
            let _ = uc.mem_write(msg_addr, msg.as_bytes());
            msg_addr
        }
        HleBuiltin::Open => {
            // int open(const char *path, int flags, mode_t mode) / __open_2(path, flags)
            let path = read_guest_cstr(uc, x0, 512);
            let (errno_addr, res) = {
                let (table, mem, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.open_with_flags(&path, x1 as u32, mem))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Close => {
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.close(x0 as i32))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Read => {
            // ssize_t read(int fd, void *buf, size_t count)
            let count = (x2 as usize).min(16 * 1024 * 1024);
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.read(x0 as i32, count))
            };
            match res {
                Ok(data) => {
                    if !data.is_empty() && x1 != 0 {
                        let _ = uc.mem_write(x1, &data);
                    }
                    data.len() as u64
                }
                Err(e) => c_errno_ret(uc, errno_addr, e),
            }
        }
        HleBuiltin::Write => {
            // ssize_t write(int fd, const void *buf, size_t count)
            let count = (x2 as usize).min(16 * 1024 * 1024);
            let mut buf = vec![0u8; count];
            if count > 0 && x1 != 0 {
                let _ = uc.mem_read(x1, &mut buf);
            }
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.write(x0 as i32, &buf))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Pread => {
            // ssize_t pread(int fd, void *buf, size_t count, off_t offset)
            let count = (x2 as usize).min(16 * 1024 * 1024);
            let offset = uc.reg_read(RegisterARM64::X3).unwrap_or(0) as usize;
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.read_at(x0 as i32, offset, count))
            };
            match res {
                Some(data) => {
                    if !data.is_empty() && x1 != 0 {
                        let _ = uc.mem_write(x1, &data);
                    }
                    data.len() as u64
                }
                None => c_errno_ret(uc, errno_addr, crate::os::vfs::EBADF),
            }
        }
        HleBuiltin::Pwrite => {
            // ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset)
            let count = (x2 as usize).min(16 * 1024 * 1024);
            let offset = uc.reg_read(RegisterARM64::X3).unwrap_or(0) as usize;
            let mut buf = vec![0u8; count];
            if count > 0 && x1 != 0 {
                let _ = uc.mem_read(x1, &mut buf);
            }
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.write_at(x0 as i32, offset, &buf))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Lseek => {
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.lseek(x0 as i32, x1 as i64, x2 as i32))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Fstat => {
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.fstat_fd(x0 as i32))
            };
            match res {
                Ok((mode, size)) => {
                    if x1 != 0 && write_aarch64_stat(uc, x1, mode, size) {
                        0
                    } else {
                        c_errno_ret(uc, errno_addr, crate::os::vfs::EINVAL)
                    }
                }
                Err(e) => c_errno_ret(uc, errno_addr, e),
            }
        }
        HleBuiltin::Stat => {
            let path = read_guest_cstr(uc, x0, 512);
            let (errno_addr, res) = {
                let (table, mem, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.stat_path(&path, mem))
            };
            match res {
                Ok((mode, size)) => {
                    if x1 != 0 && write_aarch64_stat(uc, x1, mode, size) {
                        0
                    } else {
                        c_errno_ret(uc, errno_addr, crate::os::vfs::EINVAL)
                    }
                }
                Err(e) => c_errno_ret(uc, errno_addr, e),
            }
        }
        HleBuiltin::Ftruncate => {
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.ftruncate(x0 as i32, x1))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Access => {
            let path = read_guest_cstr(uc, x0, 512);
            let (errno_addr, res) = {
                let (table, mem, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.access(&path, mem))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Mkdir => {
            let path = read_guest_cstr(uc, x0, 512);
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.mkdir(&path))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Unlink => {
            let path = read_guest_cstr(uc, x0, 512);
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.unlink(&path))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Rename => {
            let old_path = read_guest_cstr(uc, x0, 512);
            let new_path = read_guest_cstr(uc, x1, 512);
            let (errno_addr, res) = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                (table.errno_addr, vfs.rename(&old_path, &new_path))
            };
            c_errno_ret(uc, errno_addr, res)
        }
        HleBuiltin::Mmap => {
            // void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
            let hint = x0;
            let len = x1;
            let perms = linux_prot_to_uc(x2);
            let fd = uc.reg_read(RegisterARM64::X4).unwrap_or(u64::MAX) as i32;
            let offset = uc.reg_read(RegisterARM64::X5).unwrap_or(0) as usize;

            let aligned_len = align_up(len.max(PAGE_SIZE), PAGE_SIZE);
            let file_path = if fd >= 0 {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                vfs.fd_path(fd).map(str::to_string)
            } else {
                None
            };

            // MAP_SHARED 一致性：若该文件已有活跃 mmap 映射，先将其脏页刷回 VFS 再读取
            if let Some(ref target_path) = file_path {
                let existing: Vec<(u64, MmapFileBacking)> = {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    table
                        .mmap_backings
                        .iter()
                        .filter(|(_, b)| b.path == *target_path)
                        .map(|(&a, b)| (a, b.clone()))
                        .collect()
                };
                for (old_addr, b) in existing {
                    let mut buf = vec![0u8; b.len];
                    if b.len > 0 && uc.mem_read(old_addr, &mut buf).is_ok() {
                        let (_, _, vfs) = get_state(uc.get_data_mut());
                        vfs.write_at_path(&b.path, b.offset, &buf);
                    }
                }
            }

            let file_bytes = if fd >= 0 {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                vfs.read_at(fd, offset, len as usize)
            } else {
                None
            };

            let flags = uc.reg_read(RegisterARM64::X3).unwrap_or(0);
            let map_fixed = (flags & 0x10) != 0;
            let req_hint = if map_fixed && hint != 0 { hint } else { 0 };

            let label = file_path
                .as_deref()
                .map(|p| format!("[mmap:{p}]"))
                .unwrap_or_else(|| "[anon:hle_mmap]".to_string());

            let mapped_addr = {
                let mut tmp_mem =
                    std::mem::replace(get_state(uc.get_data_mut()).1, MemoryManager::new(0, 0));
                let res = tmp_mem.mmap(uc, req_hint, aligned_len, perms, label, offset as u64);
                *get_state(uc.get_data_mut()).1 = tmp_mem;
                res.ok()
            };

            if let Some(addr) = mapped_addr {
                if let Some(bytes) = file_bytes {
                    if !bytes.is_empty() {
                        let _ = uc.mem_write(addr, &bytes);
                    }
                }
                if let Some(path) = file_path {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    table.mmap_backings.insert(
                        addr,
                        MmapFileBacking {
                            path,
                            offset,
                            len: len as usize,
                        },
                    );
                }
                addr
            } else {
                u64::MAX // MAP_FAILED
            }
        }
        HleBuiltin::Msync => {
            // int msync(void *addr, size_t length, int flags)
            let backing = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table.mmap_backings.get(&x0).cloned()
            };
            if let Some(b) = backing {
                let sync_len = (x1 as usize).min(b.len);
                let mut buf = vec![0u8; sync_len];
                if sync_len > 0 && uc.mem_read(x0, &mut buf).is_ok() {
                    let (_, _, vfs) = get_state(uc.get_data_mut());
                    vfs.write_at_path(&b.path, b.offset, &buf);
                }
            }
            0
        }
        HleBuiltin::Munmap => {
            // int munmap(void *addr, size_t length)
            let backing = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table.mmap_backings.remove(&x0)
            };
            if let Some(b) = backing {
                let mut buf = vec![0u8; b.len];
                if b.len > 0 && uc.mem_read(x0, &mut buf).is_ok() {
                    let (_, _, vfs) = get_state(uc.get_data_mut());
                    vfs.write_at_path(&b.path, b.offset, &buf);
                }
            }
            let aligned_len = align_up(x1.max(PAGE_SIZE), PAGE_SIZE);
            let mut tmp_mem =
                std::mem::replace(get_state(uc.get_data_mut()).1, MemoryManager::new(0, 0));
            let _ = tmp_mem.munmap(uc, align_down(x0, PAGE_SIZE), aligned_len);
            *get_state(uc.get_data_mut()).1 = tmp_mem;
            0
        }
        HleBuiltin::Atoi => {
            let s = read_guest_cstr(uc, x0, 64);
            parse_c_i64(&s) as i32 as i64 as u64
        }
        HleBuiltin::Atol => {
            let s = read_guest_cstr(uc, x0, 128);
            parse_c_i64(&s) as u64
        }
        HleBuiltin::Sscanf => {
            // int sscanf(const char *str, const char *format, ...)
            let input = read_guest_cstr(uc, x0, 4096);
            let fmt = read_guest_cstr(uc, x1, 1024);
            run_guest_sscanf(uc, &input, &fmt) as u64
        }
        HleBuiltin::Strtol => {
            let s = read_guest_cstr(uc, x0, 128);
            let base = (x2 as u32).clamp(0, 36);
            let trimmed = s.trim();
            let parsed = if base == 0 {
                if let Some(hex) = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X"))
                {
                    i64::from_str_radix(hex, 16).unwrap_or(0)
                } else {
                    trimmed.parse::<i64>().unwrap_or(0)
                }
            } else {
                i64::from_str_radix(trimmed.trim_start_matches("0x"), base).unwrap_or(0)
            };
            if x1 != 0 {
                let end_ptr = x0 + s.len() as u64;
                let _ = uc.mem_write(x1, &end_ptr.to_le_bytes());
            }
            parsed as u64
        }
        HleBuiltin::SystemPropertyGet => {
            let key = read_guest_cstr(uc, x0, 256);
            let val = {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                vfs.get_property(&key).unwrap_or("").to_string()
            };
            let mut bytes = val.as_bytes().to_vec();
            bytes.push(0);
            let _ = uc.mem_write(x1, &bytes);
            val.len() as u64
        }
        HleBuiltin::AndroidLogPrint => {
            let tag = read_guest_cstr(uc, x1, 128);
            let fmt = read_guest_cstr(uc, x2, 512);
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.breadcrumbs()
                .push(format!("[android_log/{tag}] {fmt}"));
            tracing::info!(target: "runidbg::hle", tag = %tag, "{fmt}");
            fmt.len() as u64
        }
        HleBuiltin::Errno => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.errno_addr
        }
        HleBuiltin::Getauxval => match x0 {
            6 => PAGE_SIZE, // AT_PAGESZ
            16 | 26 => 0,   // AT_HWCAP / AT_HWCAP2（关闭可选硬件扩展以走纯兼容路径）
            other => {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                vfs.diag().record_missing_auxv(other);
                0
            }
        },
        HleBuiltin::Getpagesize => PAGE_SIZE,
        HleBuiltin::Sysconf => match x0 as i32 {
            39 | 40 => PAGE_SIZE, // Bionic _SC_PAGESIZE / _SC_PAGE_SIZE
            96 | 97 => 4,         // Bionic _SC_NPROCESSORS_CONF / _SC_NPROCESSORS_ONLN
            _ => 0,
        },
        HleBuiltin::Crc32 => {
            // uLong crc32(uLong crc, const Bytef *buf, uInt len)
            if x1 == 0 {
                0
            } else {
                let len = (x2 as usize).min(16 * 1024 * 1024);
                let mut buf = vec![0u8; len];
                if len > 0 && uc.mem_read(x1, &mut buf).is_ok() {
                    zlib_crc32(x0 as u32, &buf) as u64
                } else {
                    x0 as u32 as u64
                }
            }
        }
        HleBuiltin::Arc4Random => {
            let (tramp, _, vfs) = get_state(uc.get_data_mut());
            let mut b = [0u8; 4];
            vfs.fill_random(&mut b);
            let val = u32::from_le_bytes(b) as u64;
            tramp.push_hle_event(crate::tracer::HleOpEvent {
                kind: crate::tracer::ExtKind::HleSource,
                subop_or_byte: 1, // arc4random
                aux_subop: 0,
                dst: 0,
                src: val,
                len: 4,
                caller_pc,
            });
            val
        }
        HleBuiltin::Rand => {
            let (tramp, _, vfs) = get_state(uc.get_data_mut());
            if !tramp.rand_stub_warned {
                tramp.rand_stub_warned = true;
                tracing::warn!(
                    target: "runidbg::hle",
                    "rand()/random() 当前使用独立确定性 PRNG 桩实现（xorshift64，与 /dev/urandom 隔离，非真 glibc TYPE_3 状态机）；如需特定序列请用 emu.hook.importReplace(\"rand\", ...) 替换"
                );
            }
            let val = vfs.next_libc_rand() as u64;
            tramp.push_hle_event(crate::tracer::HleOpEvent {
                kind: crate::tracer::ExtKind::HleSource,
                subop_or_byte: 0, // rand
                aux_subop: 0,
                dst: 0,
                src: val,
                len: 4,
                caller_pc,
            });
            val
        }
        HleBuiltin::Srand => {
            let (tramp, _, vfs) = get_state(uc.get_data_mut());
            if !tramp.rand_stub_warned {
                tramp.rand_stub_warned = true;
                tracing::warn!(
                    target: "runidbg::hle",
                    "srand()/srandom() 当前作用于独立确定性 PRNG 桩实现（xorshift64，与 /dev/urandom 隔离，非真 glibc TYPE_3 状态机）；如需特定序列请用 emu.hook.importReplace(\"rand\", ...) 替换"
                );
            }
            vfs.seed_libc_rand((x0 as u32) as u64);
            0
        }
        HleBuiltin::Arc4RandomBuf => {
            let len = (x1 as usize).min(1024 * 1024);
            let mut buf = vec![0u8; len];
            let first8 = {
                let (tramp, _, vfs) = get_state(uc.get_data_mut());
                vfs.fill_random(&mut buf);
                let mut f8 = [0u8; 8];
                let n = buf.len().min(8);
                f8[..n].copy_from_slice(&buf[..n]);
                let f8_val = u64::from_le_bytes(f8);
                if x0 != 0 && !buf.is_empty() {
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleSource,
                        subop_or_byte: 4, // getrandom/arc4random_buf
                        aux_subop: 0,
                        dst: x0,
                        src: f8_val,
                        len: buf.len() as u64,
                        caller_pc,
                    });
                }
                f8_val
            };
            let _ = first8;
            if x0 != 0 && !buf.is_empty() {
                let _ = uc.mem_write(x0, &buf);
            }
            0
        }
        HleBuiltin::Getpid => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.pid as u64
        }
        HleBuiltin::Gettid | HleBuiltin::PthreadSelf => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.tid as u64
        }
        HleBuiltin::Getuid => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.uid as u64
        }
        HleBuiltin::Time => {
            let sec = {
                let (tramp, _, vfs) = get_state(uc.get_data_mut());
                let s = vfs.current_clock_ns() / 1_000_000_000;
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleSource,
                    subop_or_byte: 3, // time
                    aux_subop: 0,
                    dst: x0,
                    src: s,
                    len: if x0 != 0 { 8 } else { 0 },
                    caller_pc,
                });
                s
            };
            if x0 != 0 {
                let _ = uc.mem_write(x0, &(sec as i64).to_le_bytes());
            }
            sec
        }
        HleBuiltin::ClockGettime => {
            // int clock_gettime(clockid_t clk_id, struct timespec *tp)
            let ns = {
                let (tramp, _, vfs) = get_state(uc.get_data_mut());
                let ns = vfs.current_clock_ns();
                if x1 != 0 {
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleSource,
                        subop_or_byte: 6, // clock_gettime
                        aux_subop: 0,
                        dst: x1,
                        src: ns / 1_000_000_000,
                        len: 16,
                        caller_pc,
                    });
                }
                ns
            };
            if x1 != 0 {
                let sec = (ns / 1_000_000_000) as i64;
                let nsec = (ns % 1_000_000_000) as i64;
                let mut buf = [0u8; 16];
                buf[0..8].copy_from_slice(&sec.to_le_bytes());
                buf[8..16].copy_from_slice(&nsec.to_le_bytes());
                let _ = uc.mem_write(x1, &buf);
            }
            0
        }
        HleBuiltin::Gettimeofday => {
            // int gettimeofday(struct timeval *tv, struct timezone *tz)
            let ns = {
                let (tramp, _, vfs) = get_state(uc.get_data_mut());
                let ns = vfs.current_clock_ns();
                if x0 != 0 {
                    tramp.push_hle_event(crate::tracer::HleOpEvent {
                        kind: crate::tracer::ExtKind::HleSource,
                        subop_or_byte: 5, // gettimeofday
                        aux_subop: 0,
                        dst: x0,
                        src: ns / 1_000_000_000,
                        len: 16,
                        caller_pc,
                    });
                }
                ns
            };
            if x0 != 0 {
                let sec = (ns / 1_000_000_000) as i64;
                let usec = ((ns % 1_000_000_000) / 1_000) as i64;
                let mut buf = [0u8; 16];
                buf[0..8].copy_from_slice(&sec.to_le_bytes());
                buf[8..16].copy_from_slice(&usec.to_le_bytes());
                let _ = uc.mem_write(x0, &buf);
            }
            if x1 != 0 {
                let _ = uc.mem_write(x1, &[0u8; 8]);
            }
            0
        }
        HleBuiltin::LocaltimeR => {
            // struct tm *localtime_r(const time_t *timep, struct tm *result)
            // Linux AArch64 struct tm (56 bytes): 9 x i32 + 4B pad + i64 tm_gmtoff + u64 tm_zone
            if x1 != 0 {
                let mut t_bytes = [0u8; 8];
                let ts = if x0 != 0 && uc.mem_read(x0, &mut t_bytes).is_ok() {
                    i64::from_le_bytes(t_bytes).max(0)
                } else {
                    1_700_000_000
                };
                let sec = (ts % 60) as i32;
                let min = ((ts / 60) % 60) as i32;
                let hour = ((ts / 3600) % 24) as i32;
                let days = (ts / 86400) as i32;
                let year = 70 + (days / 365);
                let yday = days % 365;
                let mon = (yday / 30).min(11);
                let mday = (yday % 30) + 1;
                let wday = (days + 4) % 7;
                let mut tm = [0u8; 56];
                tm[0..4].copy_from_slice(&sec.to_le_bytes());
                tm[4..8].copy_from_slice(&min.to_le_bytes());
                tm[8..12].copy_from_slice(&hour.to_le_bytes());
                tm[12..16].copy_from_slice(&mday.to_le_bytes());
                tm[16..20].copy_from_slice(&mon.to_le_bytes());
                tm[20..24].copy_from_slice(&year.to_le_bytes());
                tm[24..28].copy_from_slice(&wday.to_le_bytes());
                tm[28..32].copy_from_slice(&yday.to_le_bytes());
                let _ = uc.mem_write(x1, &tm);
            }
            x1
        }
        HleBuiltin::Sleep => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.mono_clock_ns = vfs
                .mono_clock_ns
                .saturating_add(x0.saturating_mul(1_000_000_000));
            0
        }
        HleBuiltin::Usleep => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.mono_clock_ns = vfs.mono_clock_ns.saturating_add(x0.saturating_mul(1_000));
            0
        }
        HleBuiltin::Syscall => {
            // long syscall(long number, ...)
            match x0 {
                172 => {
                    let (_, _, vfs) = get_state(uc.get_data_mut());
                    vfs.pid as u64
                }
                178 => {
                    let (_, _, vfs) = get_state(uc.get_data_mut());
                    vfs.tid as u64
                }
                278 => {
                    // SYS_getrandom(buf, buflen, flags)
                    let len = (x2 as usize).min(1024 * 1024);
                    let mut buf = vec![0u8; len];
                    {
                        let (tramp, _, vfs) = get_state(uc.get_data_mut());
                        vfs.fill_random(&mut buf);
                        if x1 != 0 && !buf.is_empty() {
                            let mut f8 = [0u8; 8];
                            let n = buf.len().min(8);
                            f8[..n].copy_from_slice(&buf[..n]);
                            tramp.push_hle_event(crate::tracer::HleOpEvent {
                                kind: crate::tracer::ExtKind::HleSource,
                                subop_or_byte: 4, // getrandom
                                aux_subop: 0,
                                dst: x1,
                                src: u64::from_le_bytes(f8),
                                len: buf.len() as u64,
                                caller_pc,
                            });
                        }
                    }
                    if x1 != 0 && !buf.is_empty() {
                        let _ = uc.mem_write(x1, &buf);
                    }
                    len as u64
                }
                _ => 0,
            }
        }
        HleBuiltin::StackChkFail => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.breadcrumbs().push("[HLE WARN] __stack_chk_fail");
            tracing::warn!(target: "runidbg::hle", "触发 __stack_chk_fail (已拦截并安全返回)");
            0
        }
        HleBuiltin::Abort => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            vfs.breadcrumbs().push("[HLE FATAL] abort()");
            tracing::error!(target: "runidbg::hle", "Guest 调用 abort()，终止仿真");
            let _ = CpuBackend::stop(uc);
            0
        }
        HleBuiltin::Exit => {
            let _ = CpuBackend::stop(uc);
            x0
        }
        HleBuiltin::Memalign => {
            // void *memalign(size_t alignment, size_t size)
            let align = x0.max(16);
            let size = x1.max(1);
            let raw = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                mem.alloc_heap_aligned_with_header(size, align).unwrap_or(0)
            };
            if raw == 0 {
                0
            } else {
                let _ = uc.mem_write(raw, &0x4845_4150_424C_4B31u64.to_le_bytes());
                let _ = uc.mem_write(raw + 8, &size.to_le_bytes());
                let aligned = raw + 16;
                let (tramp, mem, _) = get_state(uc.get_data_mut());
                mem.record_heap_alloc(aligned, size, caller_pc, "memalign");
                tramp.push_hle_event(crate::tracer::HleOpEvent {
                    kind: crate::tracer::ExtKind::HleMalloc,
                    subop_or_byte: 3,
                    aux_subop: 0,
                    dst: aligned,
                    src: 0,
                    len: size,
                    caller_pc,
                });
                aligned
            }
        }
        HleBuiltin::Printf => {
            let fmt = read_guest_cstr(uc, x0, 512);
            fmt.len() as u64
        }
        HleBuiltin::Fprintf => {
            let fmt = read_guest_cstr(uc, x1, 512);
            let rendered = format_guest_printf(uc, &fmt, 2);
            let bytes = rendered.as_bytes();
            let (table, _, vfs) = get_state(uc.get_data_mut());
            if let Some(stream) = table.stdio_streams.get_mut(&x0) {
                stream.ungetc_buf = None;
                let fd = stream.fd;
                let written = vfs.write(fd, bytes);
                if written < 0 {
                    if let Some(s) = table.stdio_streams.get_mut(&x0) {
                        s.err = true;
                    }
                    (-1i64) as u64
                } else {
                    written as u64
                }
            } else {
                bytes.len() as u64
            }
        }
        HleBuiltin::Ctype(op) => {
            let c = (x0 & 0xFF) as u8;
            match op {
                "isalpha" => u64::from(c.is_ascii_alphabetic()),
                "isdigit" => u64::from(c.is_ascii_digit()),
                "isalnum" => u64::from(c.is_ascii_alphanumeric()),
                "isspace" => u64::from(c.is_ascii_whitespace()),
                "isupper" => u64::from(c.is_ascii_uppercase()),
                "islower" => u64::from(c.is_ascii_lowercase()),
                "isxdigit" => u64::from(c.is_ascii_hexdigit()),
                "isprint" => u64::from(c.is_ascii_graphic() || c == b' '),
                "toupper" => c.to_ascii_uppercase() as u64,
                "tolower" => c.to_ascii_lowercase() as u64,
                _ => 0,
            }
        }
        HleBuiltin::MathUnaryF64(op) => {
            let v = f64::from_bits(uc.reg_read(RegisterARM64::D0).unwrap_or(0));
            let res = match op {
                "sin" => v.sin(),
                "cos" => v.cos(),
                "tan" => v.tan(),
                "asin" => v.asin(),
                "acos" => v.acos(),
                "atan" => v.atan(),
                "sinh" => v.sinh(),
                "cosh" => v.cosh(),
                "tanh" => v.tanh(),
                "exp" => v.exp(),
                "exp2" => v.exp2(),
                "log" => v.ln(),
                "log2" => v.log2(),
                "log10" => v.log10(),
                "sqrt" => v.sqrt(),
                "floor" => v.floor(),
                "ceil" => v.ceil(),
                "round" => v.round(),
                "fabs" => v.abs(),
                _ => 0.0,
            };
            let _ = uc.reg_write(RegisterARM64::D0, res.to_bits());
            0
        }
        HleBuiltin::MathBinaryF64(op) => {
            let a = f64::from_bits(uc.reg_read(RegisterARM64::D0).unwrap_or(0));
            let b = f64::from_bits(uc.reg_read(RegisterARM64::D1).unwrap_or(0));
            let res = match op {
                "pow" => a.powf(b),
                "fmod" => a % b,
                "atan2" => a.atan2(b),
                "hypot" => a.hypot(b),
                _ => 0.0,
            };
            let _ = uc.reg_write(RegisterARM64::D0, res.to_bits());
            0
        }
        HleBuiltin::PthreadOnce => {
            // int pthread_once(pthread_once_t *once_control, void (*init_routine)(void))
            let mut cur = [0u8; 4];
            if x0 != 0
                && x1 != 0
                && uc.mem_read(x0, &mut cur).is_ok()
                && u32::from_le_bytes(cur) == 0
            {
                let _ = uc.mem_write(x0, &1u32.to_le_bytes());
                let _ = uc.reg_write(RegisterARM64::PC, PTHREAD_ONCE_STUB_ADDR);
                return;
            }
            0
        }
        HleBuiltin::PthreadKeyCreate => {
            let key = {
                let (table, _, _) = get_state(uc.get_data_mut());
                let k = table.next_pthread_key;
                table.next_pthread_key += 1;
                table.pthread_keys.insert(k, 0);
                k
            };
            if x0 != 0 {
                let _ = uc.mem_write(x0, &key.to_le_bytes());
            }
            0
        }
        HleBuiltin::PthreadSetSpecific => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.pthread_keys.insert(x0 as u32, x1);
            0
        }
        HleBuiltin::PthreadGetSpecific => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.pthread_keys.get(&(x0 as u32)).copied().unwrap_or(0)
        }
        HleBuiltin::Socket => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            let fd = vfs.alloc_raw_fd();
            let r = vfs.net.create_socket(fd, x0 as i32, x1 as i32, x2 as i32);
            if r < 0 { (-1i64) as u64 } else { r as u64 }
        }
        HleBuiltin::Bind => {
            let mut sa = vec![0u8; (x2 as usize).min(128)];
            if uc.mem_read(x1, &mut sa).is_err() {
                (-1i64) as u64
            } else if let Some(addr) = crate::os::net::parse_sockaddr(&sa) {
                let r = get_state(uc.get_data_mut()).2.net.bind(x0 as i32, addr);
                if r < 0 { (-1i64) as u64 } else { r as u64 }
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Listen => {
            let r = get_state(uc.get_data_mut()).2.net.listen(x0 as i32);
            if r < 0 { (-1i64) as u64 } else { r as u64 }
        }
        HleBuiltin::Accept => {
            let (new_fd, res) = {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                let nfd = vfs.alloc_raw_fd();
                let r = vfs.net.accept(x0 as i32, nfd, x3 as i32);
                (nfd, r)
            };
            match res {
                Ok(peer) => {
                    if x1 != 0 {
                        let bytes = crate::os::net::serialize_sockaddr(peer);
                        let _ = uc.mem_write(x1, &bytes);
                        if x2 != 0 {
                            let _ = uc.mem_write(x2, &(bytes.len() as u32).to_le_bytes());
                        }
                    }
                    new_fd as u64
                }
                Err(_) => (-1i64) as u64,
            }
        }
        HleBuiltin::Connect => {
            let mut sa = vec![0u8; (x2 as usize).min(128)];
            if uc.mem_read(x1, &mut sa).is_err() {
                (-1i64) as u64
            } else if let Some(addr) = crate::os::net::parse_sockaddr(&sa) {
                let r = get_state(uc.get_data_mut()).2.net.connect(x0 as i32, addr);
                if r < 0 { (-1i64) as u64 } else { r as u64 }
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Send => {
            let len = (x2 as usize).min(1024 * 1024);
            let mut data = vec![0u8; len];
            if len > 0 && uc.mem_read(x1, &mut data).is_err() {
                (-1i64) as u64
            } else {
                let r = get_state(uc.get_data_mut()).2.net.send(x0 as i32, &data, None);
                if r < 0 { (-1i64) as u64 } else { r as u64 }
            }
        }
        HleBuiltin::Sendto => {
            let len = (x2 as usize).min(1024 * 1024);
            let mut data = vec![0u8; len];
            if len > 0 && uc.mem_read(x1, &mut data).is_err() {
                (-1i64) as u64
            } else {
                let dest = if x4 != 0 && x5 >= 8 {
                    let mut sa = vec![0u8; (x5 as usize).min(128)];
                    if uc.mem_read(x4, &mut sa).is_ok() {
                        crate::os::net::parse_sockaddr(&sa)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let r = get_state(uc.get_data_mut()).2.net.send(x0 as i32, &data, dest);
                if r < 0 { (-1i64) as u64 } else { r as u64 }
            }
        }
        HleBuiltin::Recv => {
            let len = (x2 as usize).min(1024 * 1024);
            match get_state(uc.get_data_mut()).2.net.recv(x0 as i32, len) {
                Ok((data, _)) => {
                    if !data.is_empty() && uc.mem_write(x1, &data).is_err() {
                        (-1i64) as u64
                    } else {
                        data.len() as u64
                    }
                }
                Err(_) => (-1i64) as u64,
            }
        }
        HleBuiltin::Recvfrom => {
            let len = (x2 as usize).min(1024 * 1024);
            match get_state(uc.get_data_mut()).2.net.recv(x0 as i32, len) {
                Ok((data, peer)) => {
                    if !data.is_empty() && uc.mem_write(x1, &data).is_err() {
                        return;
                    }
                    if let Some(p) = peer {
                        if x4 != 0 {
                            let bytes = crate::os::net::serialize_sockaddr(p);
                            let _ = uc.mem_write(x4, &bytes);
                            if x5 != 0 {
                                let _ = uc.mem_write(x5, &(bytes.len() as u32).to_le_bytes());
                            }
                        }
                    }
                    data.len() as u64
                }
                Err(_) => (-1i64) as u64,
            }
        }
        HleBuiltin::Shutdown => {
            let r = get_state(uc.get_data_mut()).2.net.shutdown(x0 as i32, x1 as i32);
            if r < 0 { (-1i64) as u64 } else { r as u64 }
        }
        HleBuiltin::Setsockopt => 0,
        HleBuiltin::Getsockopt => {
            if x3 != 0 {
                let _ = uc.mem_write(x3, &0i32.to_le_bytes());
            }
            if x4 != 0 {
                let _ = uc.mem_write(x4, &4u32.to_le_bytes());
            }
            0
        }
        HleBuiltin::Getsockname | HleBuiltin::Getpeername => {
            let addr_opt = {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                if matches!(builtin, HleBuiltin::Getsockname) {
                    vfs.net.local_addr(x0 as i32)
                } else {
                    vfs.net.peer_addr(x0 as i32)
                }
            };
            if let Some(addr) = addr_opt {
                let bytes = crate::os::net::serialize_sockaddr(addr);
                if x1 != 0 {
                    let _ = uc.mem_write(x1, &bytes);
                }
                if x2 != 0 {
                    let _ = uc.mem_write(x2, &(bytes.len() as u32).to_le_bytes());
                }
                0
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Poll => {
            let nfds = (x1 as usize).min(1024);
            let timeout_ms = x2 as i32;
            let mut items = Vec::with_capacity(nfds);
            for i in 0..nfds {
                let mut pfd = [0u8; 8];
                if uc.mem_read(x0 + (i as u64) * 8, &mut pfd).is_ok() {
                    items.push((
                        i32::from_le_bytes(pfd[0..4].try_into().unwrap()),
                        u16::from_le_bytes(pfd[4..6].try_into().unwrap()),
                    ));
                }
            }
            let revents = get_state(uc.get_data_mut()).2.net.poll_fds(&items, timeout_ms);
            let mut count = 0u64;
            for (i, &rev) in revents.iter().enumerate() {
                if rev != 0 {
                    count += 1;
                }
                let _ = uc.mem_write(x0 + (i as u64) * 8 + 6, &rev.to_le_bytes());
            }
            count
        }
        HleBuiltin::EpollCreate1 => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            let fd = vfs.alloc_raw_fd();
            vfs.net.create_epoll(fd) as u64
        }
        HleBuiltin::EpollCtl => {
            let (ev, dt) = if x3 != 0 {
                let mut buf = [0u8; 16];
                if uc.mem_read(x3, &mut buf).is_ok() {
                    (
                        u32::from_le_bytes(buf[0..4].try_into().unwrap()),
                        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
                    )
                } else {
                    (0, 0)
                }
            } else {
                (0, 0)
            };
            let r = get_state(uc.get_data_mut())
                .2
                .net
                .epoll_ctl(x0 as i32, x1 as i32, x2 as i32, ev, dt);
            if r < 0 { (-1i64) as u64 } else { r as u64 }
        }
        HleBuiltin::EpollWait => {
            let max_ev = (x2 as i32).max(0) as usize;
            let timeout_ms = x3 as i32;
            match get_state(uc.get_data_mut())
                .2
                .net
                .epoll_wait(x0 as i32, max_ev, timeout_ms)
            {
                Ok(ready) => {
                    for (i, (ev, dt)) in ready.iter().enumerate() {
                        let mut buf = [0u8; 16];
                        buf[0..4].copy_from_slice(&ev.to_le_bytes());
                        buf[8..16].copy_from_slice(&dt.to_le_bytes());
                        let _ = uc.mem_write(x1 + (i as u64) * 16, &buf);
                    }
                    ready.len() as u64
                }
                Err(_) => (-1i64) as u64,
            }
        }
        HleBuiltin::Eventfd => {
            let (_, _, vfs) = get_state(uc.get_data_mut());
            let fd = vfs.alloc_raw_fd();
            vfs.net.create_eventfd(fd, x0, x1 as u32) as u64
        }
        HleBuiltin::Pipe => {
            let (rfd, wfd, ret) = {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                let rfd = vfs.alloc_raw_fd();
                let wfd = vfs.alloc_raw_fd();
                let ret = vfs.net.create_pipe(rfd, wfd, x1 as u32);
                (rfd, wfd, ret)
            };
            if ret == 0 && x0 != 0 {
                let mut pair = [0u8; 8];
                pair[0..4].copy_from_slice(&rfd.to_le_bytes());
                pair[4..8].copy_from_slice(&wfd.to_le_bytes());
                let _ = uc.mem_write(x0, &pair);
            }
            if ret < 0 { (-1i64) as u64 } else { 0 }
        }
        HleBuiltin::Getaddrinfo => {
            // int getaddrinfo(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res)
            let node = if x0 != 0 { read_guest_cstr(uc, x0, 256) } else { "127.0.0.1".to_string() };
            let service = if x1 != 0 { read_guest_cstr(uc, x1, 64) } else { String::new() };
            let port: u16 = service.parse().unwrap_or(match service.as_str() {
                "http" => 80,
                "https" => 443,
                _ => 0,
            });
            let resolved = get_state(uc.get_data_mut()).2.net.resolve_host(&node, port);
            match resolved {
                Ok(addrs) if !addrs.is_empty() => {
                    let first = addrs[0];
                    let sa_bytes = crate::os::net::serialize_sockaddr(first);
                    let family: i32 = match first {
                        std::net::SocketAddr::V4(_) => crate::os::net::AF_INET,
                        std::net::SocketAddr::V6(_) => crate::os::net::AF_INET6,
                    };
                    // struct addrinfo on AArch64 = 48 bytes:
                    // 0: ai_flags(i32), 4: ai_family(i32), 8: ai_socktype(i32), 12: ai_protocol(i32),
                    // 16: ai_addrlen(u32), 20: pad(4), 24: ai_addr(*mut sockaddr), 32: ai_canonname, 40: ai_next
                    let (ai_ptr, sa_ptr) = {
                        let (_, mem, _) = get_state(uc.get_data_mut());
                        let ai = mem.alloc_heap(48).unwrap_or(0);
                        let sa = mem.alloc_heap(sa_bytes.len() as u64).unwrap_or(0);
                        (ai, sa)
                    };
                    if ai_ptr != 0 && sa_ptr != 0 && x3 != 0 {
                        let _ = uc.mem_write(sa_ptr, &sa_bytes);
                        let mut ai = [0u8; 48];
                        ai[4..8].copy_from_slice(&family.to_le_bytes());
                        ai[8..12].copy_from_slice(&crate::os::net::SOCK_STREAM.to_le_bytes());
                        ai[16..20].copy_from_slice(&(sa_bytes.len() as u32).to_le_bytes());
                        ai[24..32].copy_from_slice(&sa_ptr.to_le_bytes());
                        let _ = uc.mem_write(ai_ptr, &ai);
                        let _ = uc.mem_write(x3, &ai_ptr.to_le_bytes());
                        0
                    } else {
                        (-10i64) as u64 // EAI_MEMORY
                    }
                }
                _ => (-2i64) as u64, // EAI_NONAME
            }
        }
        HleBuiltin::Freeaddrinfo => {
            if x0 != 0 {
                let sa_ptr = read_guest_u64(uc, x0 + 24);
                let (_, mem, _) = get_state(uc.get_data_mut());
                if sa_ptr != 0 {
                    mem.free_heap(sa_ptr);
                }
                mem.free_heap(x0);
            }
            0
        }
        HleBuiltin::InetPton => {
            let af = x0 as i32;
            let src = read_guest_cstr(uc, x1, 64);
            match af {
                crate::os::net::AF_INET => {
                    if let Ok(ip) = src.parse::<std::net::Ipv4Addr>() {
                        let _ = uc.mem_write(x2, &ip.octets());
                        1
                    } else {
                        0
                    }
                }
                crate::os::net::AF_INET6 => {
                    if let Ok(ip) = src.parse::<std::net::Ipv6Addr>() {
                        let _ = uc.mem_write(x2, &ip.octets());
                        1
                    } else {
                        0
                    }
                }
                _ => (-1i64) as u64,
            }
        }
        HleBuiltin::InetNtop => {
            let af = x0 as i32;
            let size = x3 as usize;
            let formatted = match af {
                crate::os::net::AF_INET => {
                    let mut oct = [0u8; 4];
                    if uc.mem_read(x1, &mut oct).is_ok() {
                        Some(std::net::Ipv4Addr::from(oct).to_string())
                    } else {
                        None
                    }
                }
                crate::os::net::AF_INET6 => {
                    let mut oct = [0u8; 16];
                    if uc.mem_read(x1, &mut oct).is_ok() {
                        Some(std::net::Ipv6Addr::from(oct).to_string())
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(s) = formatted {
                let mut bytes = s.into_bytes();
                bytes.push(0);
                if bytes.len() <= size && x2 != 0 && uc.mem_write(x2, &bytes).is_ok() {
                    x2
                } else {
                    0
                }
            } else {
                0
            }
        }
        HleBuiltin::Htons => (x0 as u16).to_be() as u64,
        HleBuiltin::Htonl => (x0 as u32).to_be() as u64,
        HleBuiltin::Getifaddrs => {
            // 返回单节点 struct ifaddrs (AArch64 = 56 bytes) 对应虚拟网卡 nic
            let nic = get_state(uc.get_data_mut()).2.net.config.nic.clone();
            let mut name_bytes = nic.name.into_bytes();
            name_bytes.push(0);
            let mut sa = [0u8; 16];
            sa[0..2].copy_from_slice(&(crate::os::net::AF_INET as u16).to_le_bytes());
            sa[4..8].copy_from_slice(&nic.ipv4);
            let mut nm = [0u8; 16];
            nm[0..2].copy_from_slice(&(crate::os::net::AF_INET as u16).to_le_bytes());
            nm[4..8].copy_from_slice(&nic.netmask);

            let (ifa_ptr, name_ptr, sa_ptr, nm_ptr) = {
                let (_, mem, _) = get_state(uc.get_data_mut());
                (
                    mem.alloc_heap(56).unwrap_or(0),
                    mem.alloc_heap(name_bytes.len() as u64).unwrap_or(0),
                    mem.alloc_heap(16).unwrap_or(0),
                    mem.alloc_heap(16).unwrap_or(0),
                )
            };
            if ifa_ptr != 0 && name_ptr != 0 && sa_ptr != 0 && nm_ptr != 0 && x0 != 0 {
                let _ = uc.mem_write(name_ptr, &name_bytes);
                let _ = uc.mem_write(sa_ptr, &sa);
                let _ = uc.mem_write(nm_ptr, &nm);
                let mut ifa = [0u8; 56];
                ifa[8..16].copy_from_slice(&name_ptr.to_le_bytes());
                ifa[16..20].copy_from_slice(&0x1043u32.to_le_bytes()); // IFF_UP|BROADCAST|RUNNING|MULTICAST
                ifa[24..32].copy_from_slice(&sa_ptr.to_le_bytes());
                ifa[32..40].copy_from_slice(&nm_ptr.to_le_bytes());
                let _ = uc.mem_write(ifa_ptr, &ifa);
                let _ = uc.mem_write(x0, &ifa_ptr.to_le_bytes());
                0
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Freeifaddrs => {
            if x0 != 0 {
                let name_ptr = read_guest_u64(uc, x0 + 8);
                let sa_ptr = read_guest_u64(uc, x0 + 24);
                let nm_ptr = read_guest_u64(uc, x0 + 32);
                let (_, mem, _) = get_state(uc.get_data_mut());
                for p in [name_ptr, sa_ptr, nm_ptr, x0] {
                    if p != 0 {
                        mem.free_heap(p);
                    }
                }
            }
            0
        }
        HleBuiltin::Fopen => {
            let path = read_guest_cstr(uc, x0, 1024);
            let mode = read_guest_cstr(uc, x1, 32);
            let (flags, append) = stdio_mode_to_flags(&mode);
            let fp = {
                let (table, mem, vfs) = get_state(uc.get_data_mut());
                let fd = vfs.open_with_flags(&path, flags, mem);
                if fd < 0 {
                    0
                } else {
                    if append {
                        let _ = vfs.lseek(fd as i32, 0, 2);
                    }
                    let handle = mem.alloc_heap(16).unwrap_or(0);
                    if handle != 0 {
                        table.stdio_streams.insert(
                            handle,
                            StdioStream {
                                fd: fd as i32,
                                eof: false,
                                err: false,
                                ungetc_buf: None,
                            },
                        );
                    }
                    handle
                }
            };
            if fp != 0 {
                let fd_val = get_state(uc.get_data_mut())
                    .0
                    .stdio_streams
                    .get(&fp)
                    .map(|s| s.fd)
                    .unwrap_or(-1);
                let _ = uc.mem_write(fp, &fd_val.to_le_bytes());
            }
            fp
        }
        HleBuiltin::Fdopen => {
            let fd = x0 as i32;
            let fp = {
                let (table, mem, _) = get_state(uc.get_data_mut());
                if fd < 0 {
                    0
                } else {
                    let handle = mem.alloc_heap(16).unwrap_or(0);
                    if handle != 0 {
                        table.stdio_streams.insert(
                            handle,
                            StdioStream {
                                fd,
                                eof: false,
                                err: false,
                                ungetc_buf: None,
                            },
                        );
                    }
                    handle
                }
            };
            if fp != 0 {
                let _ = uc.mem_write(fp, &fd.to_le_bytes());
            }
            fp
        }
        HleBuiltin::Freopen => {
            let path = read_guest_cstr(uc, x0, 1024);
            let mode = read_guest_cstr(uc, x1, 32);
            let fp = x2;
            let (flags, append) = stdio_mode_to_flags(&mode);
            let (table, mem, vfs) = get_state(uc.get_data_mut());
            if let Some(old) = table.stdio_streams.get(&fp).cloned() {
                let _ = vfs.close(old.fd);
            }
            let fd = vfs.open_with_flags(&path, flags, mem);
            if fd < 0 {
                0
            } else {
                if append {
                    let _ = vfs.lseek(fd as i32, 0, 2);
                }
                let target = if fp != 0 {
                    fp
                } else {
                    mem.alloc_heap(16).unwrap_or(0)
                };
                if target != 0 {
                    table.stdio_streams.insert(
                        target,
                        StdioStream {
                            fd: fd as i32,
                            eof: false,
                            err: false,
                            ungetc_buf: None,
                        },
                    );
                }
                target
            }
        }
        HleBuiltin::Fclose => {
            let (table, mem, vfs) = get_state(uc.get_data_mut());
            if let Some(st) = table.stdio_streams.remove(&x0) {
                let _ = vfs.close(st.fd);
                mem.free_heap(x0);
                0
            } else {
                0
            }
        }
        HleBuiltin::Fread => {
            // size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
            let dst = x0;
            let elem_size = x1 as usize;
            let nmemb = x2 as usize;
            let fp = x3;
            let total = elem_size.saturating_mul(nmemb).min(16 * 1024 * 1024);
            if elem_size == 0 || total == 0 || dst == 0 {
                0
            } else {
                let data = {
                    let (table, _, vfs) = get_state(uc.get_data_mut());
                    stdio_read_chunk(table, vfs, fp, total)
                };
                match data {
                    Some(bytes) => {
                        if !bytes.is_empty() {
                            let _ = uc.mem_write(dst, &bytes);
                        }
                        (bytes.len() / elem_size) as u64
                    }
                    None => 0,
                }
            }
        }
        HleBuiltin::Fwrite => {
            let src = x0;
            let elem_size = x1 as usize;
            let nmemb = x2 as usize;
            let fp = x3;
            let total = elem_size.saturating_mul(nmemb).min(16 * 1024 * 1024);
            if elem_size == 0 || total == 0 || src == 0 {
                0
            } else {
                let mut buf = vec![0u8; total];
                if uc.mem_read(src, &mut buf).is_err() {
                    0
                } else {
                    let (table, _, vfs) = get_state(uc.get_data_mut());
                    let fd = table.stdio_streams.get(&fp).map(|s| s.fd).unwrap_or(1);
                    let written = vfs.write(fd, &buf);
                    if written <= 0 {
                        0
                    } else {
                        (written as usize / elem_size) as u64
                    }
                }
            }
        }
        HleBuiltin::Fgets | HleBuiltin::FgetsChk => {
            // fgets(buf, size, fp)
            // Bionic: __fgets_chk(buf, size, fp, dst_len)
            // glibc:  __fgets_chk(buf, dst_len, size, fp)
            let dst = x0;
            let (max_size, fp) = if matches!(builtin, HleBuiltin::FgetsChk) {
                let (table, _, _) = get_state(uc.get_data_mut());
                if table.stdio_streams.contains_key(&x2) {
                    ((x1 as i32).max(0) as usize, x2)
                } else {
                    ((x2 as i32).max(0) as usize, x3)
                }
            } else {
                ((x1 as i32).max(0) as usize, x2)
            };
            if dst == 0 || max_size == 0 {
                0
            } else if max_size == 1 {
                let _ = uc.mem_write(dst, &[0u8]);
                dst
            } else {
                let line = {
                    let (table, _, vfs) = get_state(uc.get_data_mut());
                    stdio_read_line(table, vfs, fp, max_size - 1, b'\n')
                };
                match line {
                    Some(mut bytes) if !bytes.is_empty() => {
                        bytes.push(0);
                        let _ = uc.mem_write(dst, &bytes);
                        dst
                    }
                    _ => 0,
                }
            }
        }
        HleBuiltin::Getline | HleBuiltin::Getdelim => {
            // ssize_t getline(char **lineptr, size_t *n, FILE *stream)
            // ssize_t getdelim(char **lineptr, size_t *n, int delim, FILE *stream)
            let lineptr_ptr = x0;
            let n_ptr = x1;
            let (delim, fp) = if matches!(builtin, HleBuiltin::Getdelim) {
                (x2 as u8, x3)
            } else {
                (b'\n', x2)
            };
            let line = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                stdio_read_line(table, vfs, fp, 64 * 1024, delim)
            };
            match line {
                Some(mut bytes) if !bytes.is_empty() => {
                    let len = bytes.len();
                    bytes.push(0);
                    let cur_buf = if lineptr_ptr != 0 {
                        read_guest_u64(uc, lineptr_ptr)
                    } else {
                        0
                    };
                    let cur_cap = if n_ptr != 0 {
                        read_guest_u64(uc, n_ptr) as usize
                    } else {
                        0
                    };
                    let buf_ptr = if cur_buf != 0 && cur_cap >= bytes.len() {
                        cur_buf
                    } else {
                        let new_cap = bytes.len().max(128) as u64;
                        let new_ptr = {
                            let (_, mem, _) = get_state(uc.get_data_mut());
                            mem.alloc_heap(new_cap).unwrap_or(0)
                        };
                        if new_ptr != 0 {
                            if lineptr_ptr != 0 {
                                let _ = uc.mem_write(lineptr_ptr, &new_ptr.to_le_bytes());
                            }
                            if n_ptr != 0 {
                                let _ = uc.mem_write(n_ptr, &new_cap.to_le_bytes());
                            }
                        }
                        new_ptr
                    };
                    if buf_ptr != 0 {
                        let _ = uc.mem_write(buf_ptr, &bytes);
                        len as u64
                    } else {
                        (-1i64) as u64
                    }
                }
                _ => (-1i64) as u64,
            }
        }
        HleBuiltin::Fputs => {
            let s = read_guest_cstr(uc, x0, 65536);
            let fp = x1;
            let (table, _, vfs) = get_state(uc.get_data_mut());
            let fd = table.stdio_streams.get(&fp).map(|st| st.fd).unwrap_or(1);
            let w = vfs.write(fd, s.as_bytes());
            if w < 0 { (-1i64) as u64 } else { w as u64 }
        }
        HleBuiltin::Fgetc => {
            let (table, _, vfs) = get_state(uc.get_data_mut());
            match stdio_read_chunk(table, vfs, x0, 1) {
                Some(b) if !b.is_empty() => b[0] as u64,
                _ => (-1i64) as u64,
            }
        }
        HleBuiltin::Fputc => {
            let ch = x0 as u8;
            let fp = x1;
            let (table, _, vfs) = get_state(uc.get_data_mut());
            let fd = table.stdio_streams.get(&fp).map(|st| st.fd).unwrap_or(1);
            if vfs.write(fd, &[ch]) == 1 {
                ch as u64
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Ungetc => {
            let ch = x0 as i32;
            let fp = x1;
            if ch < 0 {
                (-1i64) as u64
            } else {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(st) = table.stdio_streams.get_mut(&fp) {
                    st.ungetc_buf = Some(ch as u8);
                    st.eof = false;
                    (ch as u8) as u64
                } else {
                    (-1i64) as u64
                }
            }
        }
        HleBuiltin::Fseek => {
            let fp = x0;
            let offset = x1 as i64;
            let whence = x2 as i32;
            let (table, _, vfs) = get_state(uc.get_data_mut());
            if let Some(st) = table.stdio_streams.get_mut(&fp) {
                st.ungetc_buf = None;
                st.eof = false;
                if vfs.lseek(st.fd, offset, whence) >= 0 {
                    0
                } else {
                    (-1i64) as u64
                }
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Ftell => {
            let (table, _, vfs) = get_state(uc.get_data_mut());
            if let Some(st) = table.stdio_streams.get(&x0) {
                let pos = vfs.lseek(st.fd, 0, 1);
                if pos < 0 {
                    (-1i64) as u64
                } else if st.ungetc_buf.is_some() && pos > 0 {
                    (pos - 1) as u64
                } else {
                    pos as u64
                }
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::Rewind => {
            let (table, _, vfs) = get_state(uc.get_data_mut());
            if let Some(st) = table.stdio_streams.get_mut(&x0) {
                st.ungetc_buf = None;
                st.eof = false;
                st.err = false;
                let _ = vfs.lseek(st.fd, 0, 0);
            }
            0
        }
        HleBuiltin::Feof => {
            let (table, _, _) = get_state(uc.get_data_mut());
            u64::from(table.stdio_streams.get(&x0).is_some_and(|s| s.eof))
        }
        HleBuiltin::Ferror => {
            let (table, _, _) = get_state(uc.get_data_mut());
            u64::from(table.stdio_streams.get(&x0).is_some_and(|s| s.err))
        }
        HleBuiltin::Clearerr => {
            let (table, _, _) = get_state(uc.get_data_mut());
            if let Some(st) = table.stdio_streams.get_mut(&x0) {
                st.eof = false;
                st.err = false;
            }
            0
        }
        HleBuiltin::Fflush => 0,
        HleBuiltin::Fileno => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table
                .stdio_streams
                .get(&x0)
                .map(|s| s.fd as i64 as u64)
                .unwrap_or((-1i64) as u64)
        }
        HleBuiltin::Fscanf => {
            // int fscanf(FILE *stream, const char *format, ...)
            let fp = x0;
            let fmt = read_guest_cstr(uc, x1, 1024);
            let chunk = {
                let (table, _, vfs) = get_state(uc.get_data_mut());
                stdio_read_chunk(table, vfs, fp, 4096)
            };
            match chunk {
                Some(bytes) if !bytes.is_empty() => {
                    let text = String::from_utf8_lossy(&bytes);
                    let (matched, consumed) = run_guest_scanf_core(uc, &text, &fmt);
                    let excess = bytes.len().saturating_sub(consumed);
                    if excess > 0 {
                        let (table, _, vfs) = get_state(uc.get_data_mut());
                        stdio_seek_back(table, vfs, fp, excess);
                    }
                    matched as i64 as u64
                }
                _ => (-1i64) as u64, // EOF
            }
        }
        HleBuiltin::Lrand48 => {
            let (table, _, _) = get_state(uc.get_data_mut());
            let val = (table.next_lcg48() >> 17) & 0x7FFF_FFFF;
            table.push_hle_event(crate::tracer::HleOpEvent {
                kind: crate::tracer::ExtKind::HleSource,
                subop_or_byte: 2, // lrand48
                aux_subop: 0,
                dst: 0,
                src: val,
                len: 4,
                caller_pc,
            });
            val
        }
        HleBuiltin::Mrand48 => {
            let (table, _, _) = get_state(uc.get_data_mut());
            ((table.next_lcg48() >> 16) as u32 as i32) as i64 as u64
        }
        HleBuiltin::Srand48 => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.lcg48_state = ((x0 & 0xFFFF_FFFF) << 16) | 0x330E;
            0
        }
        HleBuiltin::Drand48 => {
            let (table, _, _) = get_state(uc.get_data_mut());
            let val = (table.next_lcg48() as f64) / ((1u64 << 48) as f64);
            let bits = val.to_bits();
            let _ = uc.reg_write(RegisterARM64::D0, bits);
            bits
        }
        // ---- 动态链接器运行时 API (dlopen / dlsym / dlclose / dlerror / dladdr / dl_iterate_phdr) ----
        HleBuiltin::Dlopen => {
            // void* dlopen(const char* filename, int flags)
            const DL_HANDLE_MAIN: u64 = 0xD100_0000;
            const DL_HANDLE_SYS: u64 = 0xD1FF_0001;
            if x0 == 0 {
                DL_HANDLE_MAIN
            } else {
                let raw = read_guest_cstr(uc, x0, 512);
                let base = raw.rsplit('/').next().unwrap_or(&raw);
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(idx) = table
                    .loaded_modules
                    .iter()
                    .position(|m| m.name == raw || m.name == base)
                {
                    table.last_dlerror = None;
                    DL_HANDLE_MAIN + 1 + (idx as u64)
                } else if matches!(
                    base,
                    "libc.so"
                        | "libm.so"
                        | "libdl.so"
                        | "liblog.so"
                        | "libz.so"
                        | "libandroid.so"
                        | "libc++.so"
                        | "libc++_shared.so"
                        | "libstdc++.so"
                        | "libnativewindow.so"
                ) {
                    table.last_dlerror = None;
                    DL_HANDLE_SYS
                } else {
                    table.last_dlerror = Some(format!("library \"{raw}\" not found"));
                    0
                }
            }
        }
        HleBuiltin::Dlsym => {
            // void* dlsym(void* handle, const char* symbol)
            const DL_HANDLE_MAIN: u64 = 0xD100_0000;
            const DL_HANDLE_SYS: u64 = 0xD1FF_0001;
            let sym = read_guest_cstr(uc, x1, 256);
            if sym.is_empty() {
                0
            } else {
                // 1. 先按句柄在已加载 ELF 模块中查找
                let found_in_mod = {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    if x0 > DL_HANDLE_MAIN && x0 < DL_HANDLE_SYS {
                        let idx = (x0 - DL_HANDLE_MAIN - 1) as usize;
                        table
                            .loaded_modules
                            .get(idx)
                            .and_then(|m| m.symbols.get(&sym).copied())
                    } else if x0 == 0 || x0 == DL_HANDLE_MAIN || x0 == (-1i64) as u64 {
                        table
                            .loaded_modules
                            .iter()
                            .find_map(|m| m.symbols.get(&sym).copied())
                    } else {
                        None
                    }
                };
                if let Some(addr) = found_in_mod {
                    addr
                } else if x0 == 0
                    || x0 == DL_HANDLE_MAIN
                    || x0 == DL_HANDLE_SYS
                    || x0 == (-1i64) as u64
                {
                    // 2. 回退到内置 HLE 跳板表（返回真实可调用的 ARM64 跳板地址）
                    if HleBuiltin::from_symbol(&sym).is_some() {
                        let existing = get_state(uc.get_data_mut()).0.get_addr(&sym);
                        if let Some(addr) = existing {
                            addr
                        } else {
                            // 动态分配新跳板槽位
                            let (addr, ok) = {
                                let (table, mem, _) = get_state(uc.get_data_mut());
                                let slot_idx = table.slots.len() as u64;
                                let a = TRAMPOLINE_BASE + slot_idx * 4;
                                let action = HleBuiltin::from_symbol(&sym).unwrap();
                                table.slots.push(action);
                                table.name_to_addr.insert(sym.clone(), a);
                                let _ = mem;
                                (a, true)
                            };
                            if ok {
                                let insn = if uc.uses_svc_trampolines() {
                                    ARM64_SVC_HLE_INSN
                                } else {
                                    ARM64_RET_INSN
                                };
                                let _ = uc.mem_write(addr, &insn.to_le_bytes());
                                let _ = uc.flush_cache(addr, addr + 4);
                            }
                            addr
                        }
                    } else {
                        let (table, _, _) = get_state(uc.get_data_mut());
                        table.last_dlerror = Some(format!("undefined symbol: {sym}"));
                        0
                    }
                } else {
                    0
                }
            }
        }
        HleBuiltin::Dlclose => 0,
        HleBuiltin::Dlerror => {
            let err_str = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table.last_dlerror.take()
            };
            if let Some(msg) = err_str {
                let mut bytes = msg.into_bytes();
                bytes.push(0);
                let ptr = {
                    let (_, mem, _) = get_state(uc.get_data_mut());
                    mem.alloc_heap(bytes.len() as u64).unwrap_or(0)
                };
                if ptr != 0 {
                    let _ = uc.mem_write(ptr, &bytes);
                }
                ptr
            } else {
                0
            }
        }
        HleBuiltin::Dladdr => {
            // int dladdr(const void* addr, Dl_info* info)
            // ARM64 Dl_info (32 bytes): dli_fname(0), dli_fbase(8), dli_sname(16), dli_saddr(24)
            if x1 == 0 {
                0
            } else {
                let found = {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    table
                        .loaded_modules
                        .iter()
                        .find(|m| x0 >= m.base_addr && x0 < m.base_addr + m.image_size)
                        .map(|m| {
                            let best_sym = m
                                .symbols
                                .iter()
                                .filter(|(_, &saddr)| saddr <= x0)
                                .max_by_key(|(_, &saddr)| saddr)
                                .map(|(sname, &saddr)| (sname.clone(), saddr));
                            (m.name.clone(), m.base_addr, best_sym)
                        })
                };
                if let Some((mod_name, fbase, best_sym)) = found {
                    let mut fname_bytes = mod_name.into_bytes();
                    fname_bytes.push(0);
                    let (fname_ptr, sname_ptr, saddr) = {
                        let (_, mem, _) = get_state(uc.get_data_mut());
                        let fp = mem.alloc_heap(fname_bytes.len() as u64).unwrap_or(0);
                        let (sp, sa) = if let Some((sn, sa)) = best_sym {
                            let sp = mem.alloc_heap((sn.len() + 1) as u64).unwrap_or(0);
                            (Some((sp, sn)), sa)
                        } else {
                            (None, 0)
                        };
                        (fp, sp, sa)
                    };
                    if fname_ptr != 0 {
                        let _ = uc.mem_write(fname_ptr, &fname_bytes);
                    }
                    let sname_addr = if let Some((sp, sn)) = sname_ptr {
                        if sp != 0 {
                            let mut sb = sn.into_bytes();
                            sb.push(0);
                            let _ = uc.mem_write(sp, &sb);
                        }
                        sp
                    } else {
                        0
                    };
                    let mut info = [0u8; 32];
                    info[0..8].copy_from_slice(&fname_ptr.to_le_bytes());
                    info[8..16].copy_from_slice(&fbase.to_le_bytes());
                    info[16..24].copy_from_slice(&sname_addr.to_le_bytes());
                    info[24..32].copy_from_slice(&saddr.to_le_bytes());
                    let _ = uc.mem_write(x1, &info);
                    1
                } else {
                    0
                }
            }
        }
        HleBuiltin::DlIteratePhdr => {
            // int dl_iterate_phdr(int (*cb)(struct dl_phdr_info*, size_t, void*), void* data)
            if x0 == 0 {
                0
            } else {
                let mods: Vec<(String, u64)> = {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    table
                        .loaded_modules
                        .iter()
                        .map(|m| (m.name.clone(), m.base_addr))
                        .collect()
                };
                if mods.is_empty() {
                    0
                } else {
                    let count = mods.len();
                    let arr_ptr = {
                        let (_, mem, _) = get_state(uc.get_data_mut());
                        mem.alloc_heap((count * 64) as u64).unwrap_or(0)
                    };
                    if arr_ptr == 0 {
                        0
                    } else {
                        for (i, (name, base)) in mods.into_iter().enumerate() {
                            let mut nb = name.into_bytes();
                            nb.push(0);
                            let nptr = {
                                let (_, mem, _) = get_state(uc.get_data_mut());
                                mem.alloc_heap(nb.len() as u64).unwrap_or(0)
                            };
                            if nptr != 0 {
                                let _ = uc.mem_write(nptr, &nb);
                            }
                            // ARM64 struct dl_phdr_info: dlpi_addr @ +0, dlpi_name @ +8, dlpi_phdr @ +16, dlpi_phnum @ +24
                            let mut entry = [0u8; 64];
                            entry[0..8].copy_from_slice(&base.to_le_bytes());
                            entry[8..16].copy_from_slice(&nptr.to_le_bytes());
                            let _ = uc.mem_write(arr_ptr + (i as u64) * 64, &entry);
                        }
                        let _ = uc.reg_write(RegisterARM64::X2, arr_ptr);
                        let _ = uc.reg_write(RegisterARM64::X3, count as u64);
                        let _ = uc.reg_write(RegisterARM64::PC, DL_ITERATE_STUB_ADDR);
                        return;
                    }
                }
            }
        }
        // ---- 目录遍历 API (opendir / readdir / closedir / rewinddir) ----
        HleBuiltin::Opendir => {
            let path = read_guest_cstr(uc, x0, 512);
            let entries = {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                vfs.list_dir(&path)
            };
            if let Some(entries) = entries {
                let (dir_ptr, dirent_buf) = {
                    let (table, mem, _) = get_state(uc.get_data_mut());
                    let dp = mem.alloc_heap(32).unwrap_or(0);
                    let db = mem.alloc_heap(304).unwrap_or(0);
                    if dp != 0 && db != 0 {
                        table.dir_streams.insert(
                            dp,
                            DirStream {
                                entries,
                                cursor: 0,
                                dirent_buf: db,
                            },
                        );
                    }
                    (dp, db)
                };
                if dir_ptr != 0 && dirent_buf != 0 {
                    dir_ptr
                } else {
                    0
                }
            } else {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                vfs.diag().record_missing_file(&path);
                0
            }
        }
        HleBuiltin::Readdir => {
            let next_item = {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(ds) = table.dir_streams.get_mut(&x0) {
                    if ds.cursor < ds.entries.len() {
                        let idx = ds.cursor;
                        ds.cursor += 1;
                        let (ref name, dtype) = ds.entries[idx];
                        Some((ds.dirent_buf, idx as u64 + 1, dtype, name.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some((buf, ino, dtype, name)) = next_item {
                // ARM64 Bionic struct dirent: u64 d_ino @ 0, i64 d_off @ 8, u16 d_reclen @ 16, u8 d_type @ 18, char d_name[256] @ 19
                let mut raw = [0u8; 280];
                let name_bytes = name.as_bytes();
                let copy_len = name_bytes.len().min(255);
                let reclen = ((19 + copy_len + 1 + 7) & !7) as u16;
                raw[0..8].copy_from_slice(&ino.to_le_bytes());
                raw[8..16].copy_from_slice(&(ino as i64).to_le_bytes());
                raw[16..18].copy_from_slice(&reclen.to_le_bytes());
                raw[18] = dtype;
                raw[19..19 + copy_len].copy_from_slice(&name_bytes[..copy_len]);
                raw[19 + copy_len] = 0;
                let _ = uc.mem_write(buf, &raw);
                buf
            } else {
                0
            }
        }
        HleBuiltin::Closedir => {
            let removed = {
                let (table, mem, _) = get_state(uc.get_data_mut());
                if let Some(ds) = table.dir_streams.remove(&x0) {
                    mem.free_heap(ds.dirent_buf);
                    mem.free_heap(x0);
                    true
                } else {
                    false
                }
            };
            if removed { 0 } else { (-1i64) as u64 }
        }
        HleBuiltin::Rewinddir => {
            let (table, _, _) = get_state(uc.get_data_mut());
            if let Some(ds) = table.dir_streams.get_mut(&x0) {
                ds.cursor = 0;
            }
            0
        }
        // ---- Bionic 新版属性 API (__system_property_find / read / read_callback) ----
        HleBuiltin::SystemPropertyFind => {
            let key = read_guest_cstr(uc, x0, 256);
            let val = {
                let (_, _, vfs) = get_state(uc.get_data_mut());
                let v = vfs.get_property(&key).map(str::to_string);
                if v.is_none() {
                    vfs.diag().record_missing_property(&key);
                }
                v
            };
            if let Some(value) = val {
                let (pi, nptr, vptr) = {
                    let (table, mem, _) = get_state(uc.get_data_mut());
                    let pi = mem.alloc_heap(16).unwrap_or(0);
                    let np = mem.alloc_heap((key.len() + 1) as u64).unwrap_or(0);
                    let vp = mem.alloc_heap((value.len() + 1) as u64).unwrap_or(0);
                    if pi != 0 {
                        table.prop_infos.insert(
                            pi,
                            PropInfoEntry {
                                name: key.clone(),
                                value: value.clone(),
                                name_ptr: np,
                                value_ptr: vp,
                            },
                        );
                    }
                    (pi, np, vp)
                };
                if nptr != 0 {
                    let mut nb = key.into_bytes();
                    nb.push(0);
                    let _ = uc.mem_write(nptr, &nb);
                }
                if vptr != 0 {
                    let mut vb = value.into_bytes();
                    vb.push(0);
                    let _ = uc.mem_write(vptr, &vb);
                }
                pi
            } else {
                0
            }
        }
        HleBuiltin::SystemPropertyRead => {
            // int __system_property_read(const prop_info* pi, char* name, char* value)
            let entry = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table
                    .prop_infos
                    .get(&x0)
                    .map(|p| (p.name.clone(), p.value.clone()))
            };
            if let Some((name, val)) = entry {
                if x1 != 0 {
                    let mut nb = name.into_bytes();
                    nb.push(0);
                    let _ = uc.mem_write(x1, &nb);
                }
                if x2 != 0 {
                    let mut vb = val.as_bytes().to_vec();
                    vb.push(0);
                    let _ = uc.mem_write(x2, &vb);
                }
                val.len() as u64
            } else {
                0
            }
        }
        HleBuiltin::SystemPropertyReadCallback => {
            // void __system_property_read_callback(const prop_info* pi, void (*cb)(void* cookie, const char* name, const char* value, uint32_t serial), void* cookie)
            if x1 == 0 {
                0
            } else {
                let ptrs = {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    table
                        .prop_infos
                        .get(&x0)
                        .map(|p| (p.name_ptr, p.value_ptr))
                };
                if let Some((nptr, vptr)) = ptrs {
                    let _ = uc.reg_write(RegisterARM64::X9, x1); // callback
                    let _ = uc.reg_write(RegisterARM64::X0, x2); // cookie
                    let _ = uc.reg_write(RegisterARM64::X1, nptr); // name
                    let _ = uc.reg_write(RegisterARM64::X2, vptr); // value
                    let _ = uc.reg_write(RegisterARM64::X3, 1); // serial
                    let _ = uc.reg_write(RegisterARM64::PC, PROP_CALLBACK_STUB_ADDR);
                    return;
                }
                0
            }
        }
        HleBuiltin::PthreadCreate => {
            // int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg)
            // 延迟到 pthread_join 时协作执行：
            // 1) 后台守护线程（while(1) sleep 心跳/反调试线程，永不 join）不会阻塞主线程导致死循环；
            // 2) 需要取回结果的工作线程在 pthread_join(tid, &retval) 时执行 start_routine(arg) 并写回 retval。
            let tid_ptr = x0;
            let start_routine = x2;
            let arg = x3;
            let tid = {
                let (table, _, _) = get_state(uc.get_data_mut());
                let tid = table.next_pthread_tid;
                table.next_pthread_tid += 1;
                if start_routine != 0 {
                    table.pending_threads.insert(tid, (start_routine, arg));
                }
                tid
            };
            if tid_ptr != 0 {
                let _ = uc.mem_write(tid_ptr, &tid.to_le_bytes());
            }
            0
        }
        HleBuiltin::PthreadJoin => {
            // int pthread_join(pthread_t thread, void **retval)
            let cur_sp = uc.reg_read(RegisterARM64::SP).unwrap_or(0);
            let pending = {
                let (table, _, _) = get_state(uc.get_data_mut());
                while let Some(&(frame_sp, _)) = table.pthread_exit_stack.last() {
                    if frame_sp < cur_sp {
                        table.pthread_exit_stack.pop();
                    } else {
                        break;
                    }
                }
                let dummy_slot = table.errno_addr + 8;
                let retval_slot = if x1 != 0 { x1 } else { dummy_slot };
                if let Some((start_routine, arg)) = table.pending_threads.remove(&x0) {
                    table
                        .pthread_exit_stack
                        .push((cur_sp.wrapping_sub(32), retval_slot));
                    Some((start_routine, arg, retval_slot))
                } else {
                    None
                }
            };
            if let Some((start_routine, arg, retval_slot)) = pending {
                let _ = uc.reg_write(RegisterARM64::X0, arg);
                let _ = uc.reg_write(RegisterARM64::X1, start_routine);
                let _ = uc.reg_write(RegisterARM64::X2, retval_slot);
                let _ = uc.reg_write(RegisterARM64::PC, PTHREAD_CREATE_STUB_ADDR);
                return;
            }
            if x1 != 0 {
                let _ = uc.mem_write(x1, &0u64.to_le_bytes());
            }
            0
        }
        HleBuiltin::PthreadDetach => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.pending_threads.remove(&x0);
            0
        }
        HleBuiltin::PthreadExit => {
            // void pthread_exit(void *retval)
            let cur_sp = uc.reg_read(RegisterARM64::SP).unwrap_or(0);
            let frame = {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(&(frame_sp, _)) = table.pthread_exit_stack.last() {
                    if cur_sp <= frame_sp {
                        table.pthread_exit_stack.pop()
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some((frame_sp, retval_slot)) = frame {
                let _ = uc.reg_write(RegisterARM64::SP, frame_sp);
                let _ = uc.reg_write(RegisterARM64::X19, retval_slot);
                let _ = uc.reg_write(RegisterARM64::X0, x0);
                let _ = uc.reg_write(RegisterARM64::PC, PTHREAD_CREATE_STUB_ADDR + 16);
                return;
            }
            let _ = CpuBackend::stop(uc);
            return;
        }
        HleBuiltin::Setjmp => {
            // int setjmp(jmp_buf env) / int sigsetjmp(sigjmp_buf env, int savesigs)
            if x0 != 0 {
                let mut buf = [0u8; 21 * 8];
                for (i, &reg) in JMPBUF_REGS.iter().enumerate() {
                    let v = uc.reg_read(reg).unwrap_or(0);
                    buf[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
                }
                let _ = uc.mem_write(x0, &buf);
            }
            0
        }
        HleBuiltin::Longjmp => {
            // void longjmp(jmp_buf env, int val) / void siglongjmp(sigjmp_buf env, int val)
            if x0 != 0 {
                let mut buf = [0u8; 21 * 8];
                if uc.mem_read(x0, &mut buf).is_ok() {
                    for (i, &reg) in JMPBUF_REGS.iter().enumerate() {
                        let v = u64::from_le_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap());
                        let _ = uc.reg_write(reg, v);
                    }
                    let target_pc = u64::from_le_bytes(buf[11 * 8..12 * 8].try_into().unwrap());
                    let val = if (x1 as i32) == 0 {
                        1u64
                    } else {
                        (x1 as i32) as i64 as u64
                    };
                    let _ = uc.reg_write(RegisterARM64::X0, val);
                    let _ = uc.reg_write(RegisterARM64::PC, target_pc);
                    return;
                }
            }
            0
        }
        HleBuiltin::Sigaction => {
            // int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact)
            let signum = x0 as i32;
            let prev = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table.sigactions.get(&signum).copied().unwrap_or([0u8; 32])
            };
            if x2 != 0 {
                let _ = uc.mem_write(x2, &prev);
            }
            if x1 != 0 {
                let mut act = [0u8; 32];
                if uc.mem_read(x1, &mut act).is_ok() {
                    let (table, _, _) = get_state(uc.get_data_mut());
                    table.sigactions.insert(signum, act);
                }
            }
            0
        }
        HleBuiltin::Signal => {
            // sighandler_t signal(int signum, sighandler_t handler)
            let signum = x0 as i32;
            let (table, _, _) = get_state(uc.get_data_mut());
            let prev = table
                .sigactions
                .get(&signum)
                .map(|a| u64::from_le_bytes(a[0..8].try_into().unwrap()))
                .unwrap_or(0);
            let mut next = [0u8; 32];
            next[0..8].copy_from_slice(&x1.to_le_bytes());
            table.sigactions.insert(signum, next);
            prev
        }
        HleBuiltin::Sigprocmask => {
            // int sigprocmask(int how, const sigset_t *set, sigset_t *oldset)
            let old = {
                let (table, _, _) = get_state(uc.get_data_mut());
                table.sigmask
            };
            if x2 != 0 {
                let _ = uc.mem_write(x2, &old.to_le_bytes());
            }
            if x1 != 0 {
                let new_set = read_guest_u64(uc, x1);
                let (table, _, _) = get_state(uc.get_data_mut());
                match x0 as i32 {
                    0 => table.sigmask |= new_set,  // SIG_BLOCK
                    1 => table.sigmask &= !new_set, // SIG_UNBLOCK
                    2 => table.sigmask = new_set,   // SIG_SETMASK
                    _ => {}
                }
            }
            0
        }
        HleBuiltin::Sigemptyset => {
            if x0 != 0 {
                let _ = uc.mem_write(x0, &0u64.to_le_bytes());
            }
            0
        }
        HleBuiltin::Sigfillset => {
            if x0 != 0 {
                let _ = uc.mem_write(x0, &(!0u64).to_le_bytes());
            }
            0
        }
        HleBuiltin::Sigaddset => {
            let sig = x1 as i32;
            if x0 != 0 && (1..=64).contains(&sig) {
                let cur = read_guest_u64(uc, x0);
                let _ = uc.mem_write(x0, &(cur | (1u64 << (sig - 1))).to_le_bytes());
            }
            0
        }
        HleBuiltin::Sigdelset => {
            let sig = x1 as i32;
            if x0 != 0 && (1..=64).contains(&sig) {
                let cur = read_guest_u64(uc, x0);
                let _ = uc.mem_write(x0, &(cur & !(1u64 << (sig - 1))).to_le_bytes());
            }
            0
        }
        HleBuiltin::Sigismember => {
            let sig = x1 as i32;
            if x0 != 0 && (1..=64).contains(&sig) {
                let cur = read_guest_u64(uc, x0);
                u64::from((cur & (1u64 << (sig - 1))) != 0)
            } else {
                0
            }
        }
        HleBuiltin::AAssetManagerFromJava => {
            // AAssetManager* AAssetManager_fromJava(JNIEnv* env, jobject assetManager)
            0xAA55_0001
        }
        HleBuiltin::AAssetManagerOpen => {
            // AAsset* AAssetManager_open(AAssetManager* mgr, const char* filename, int mode)
            let name = read_guest_cstr(uc, x1, 512);
            let trimmed = name.trim_start_matches('/');
            let candidates = [
                format!("/assets/{trimmed}"),
                format!("/{trimmed}"),
                name.clone(),
            ];
            let (table, mem, vfs) = get_state(uc.get_data_mut());
            let mut found = None;
            for cand in &candidates {
                let fd = vfs.open(cand, mem);
                if fd >= 0 {
                    let fd = fd as i32;
                    let mut all = Vec::new();
                    loop {
                        match vfs.read(fd, 65536) {
                            Ok(chunk) if !chunk.is_empty() => all.extend_from_slice(&chunk),
                            _ => break,
                        }
                    }
                    let _ = vfs.close(fd);
                    found = Some(all);
                    break;
                }
            }
            if let Some(data) = found {
                if let Ok(handle) = mem.alloc_heap(8) {
                    table.asset_streams.insert(
                        handle,
                        AssetStream {
                            data,
                            pos: 0,
                            guest_buf: None,
                        },
                    );
                    handle
                } else {
                    0
                }
            } else {
                0
            }
        }
        HleBuiltin::AAssetManagerOpenDir => {
            // AAssetDir* AAssetManager_openDir(AAssetManager* mgr, const char* dirName)
            let dir_name = read_guest_cstr(uc, x1, 512);
            let trimmed = dir_name.trim_matches('/');
            let vfs_dir = if trimmed.is_empty() {
                "/assets".to_string()
            } else {
                format!("/assets/{trimmed}")
            };
            let (table, mem, vfs) = get_state(uc.get_data_mut());
            let files: Vec<String> = vfs
                .list_dir(&vfs_dir)
                .unwrap_or_default()
                .into_iter()
                .filter(|(n, d_type)| n != "." && n != ".." && *d_type != 4)
                .map(|(n, _)| n)
                .collect();
            if let (Ok(handle), Ok(name_buf)) = (mem.alloc_heap(8), mem.alloc_heap(512)) {
                table.asset_dirs.insert(
                    handle,
                    AssetDirStream {
                        files,
                        cursor: 0,
                        name_buf,
                    },
                );
                handle
            } else {
                0
            }
        }
        HleBuiltin::AAssetRead => {
            // int AAsset_read(AAsset* asset, void* buf, size_t count)
            let chunk = {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(st) = table.asset_streams.get_mut(&x0) {
                    let avail = st.data.len().saturating_sub(st.pos);
                    let n = (x2 as usize).min(avail);
                    let slice = st.data[st.pos..st.pos + n].to_vec();
                    st.pos += n;
                    Some(slice)
                } else {
                    None
                }
            };
            if let Some(bytes) = chunk {
                if !bytes.is_empty() && x1 != 0 {
                    let _ = uc.mem_write(x1, &bytes);
                }
                bytes.len() as u64
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::AAssetSeek => {
            // off_t AAsset_seek(AAsset* asset, off_t offset, int whence)
            let (table, _, _) = get_state(uc.get_data_mut());
            if let Some(st) = table.asset_streams.get_mut(&x0) {
                let off = x1 as i64;
                let base = match x2 as i32 {
                    0 => 0i64,
                    1 => st.pos as i64,
                    2 => st.data.len() as i64,
                    _ => return,
                };
                let next = base.saturating_add(off).clamp(0, st.data.len() as i64) as usize;
                st.pos = next;
                next as u64
            } else {
                (-1i64) as u64
            }
        }
        HleBuiltin::AAssetGetLength => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table
                .asset_streams
                .get(&x0)
                .map(|st| st.data.len() as u64)
                .unwrap_or(0)
        }
        HleBuiltin::AAssetGetRemainingLength => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table
                .asset_streams
                .get(&x0)
                .map(|st| st.data.len().saturating_sub(st.pos) as u64)
                .unwrap_or(0)
        }
        HleBuiltin::AAssetGetBuffer => {
            let existing_or_alloc = {
                let (table, mem, _) = get_state(uc.get_data_mut());
                if let Some(st) = table.asset_streams.get_mut(&x0) {
                    if let Some(ptr) = st.guest_buf {
                        Some((ptr, None))
                    } else if let Ok(ptr) = mem.alloc_heap(st.data.len().max(1) as u64) {
                        st.guest_buf = Some(ptr);
                        Some((ptr, Some(st.data.clone())))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            match existing_or_alloc {
                Some((ptr, Some(bytes))) => {
                    if !bytes.is_empty() {
                        let _ = uc.mem_write(ptr, &bytes);
                    }
                    ptr
                }
                Some((ptr, None)) => ptr,
                None => 0,
            }
        }
        HleBuiltin::AAssetClose => {
            let (table, mem, _) = get_state(uc.get_data_mut());
            if let Some(st) = table.asset_streams.remove(&x0) {
                if let Some(ptr) = st.guest_buf {
                    mem.free_heap(ptr);
                }
            }
            mem.free_heap(x0);
            0
        }
        HleBuiltin::AAssetDirGetNextFileName => {
            let next = {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(dir) = table.asset_dirs.get_mut(&x0) {
                    if dir.cursor < dir.files.len() {
                        let name = dir.files[dir.cursor].clone();
                        dir.cursor += 1;
                        Some((dir.name_buf, name))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some((buf, name)) = next {
                let mut bytes = name.into_bytes();
                bytes.truncate(511);
                bytes.push(0);
                let _ = uc.mem_write(buf, &bytes);
                buf
            } else {
                0
            }
        }
        HleBuiltin::AAssetDirRewind => {
            let (table, _, _) = get_state(uc.get_data_mut());
            if let Some(dir) = table.asset_dirs.get_mut(&x0) {
                dir.cursor = 0;
            }
            0
        }
        HleBuiltin::AAssetDirClose => {
            let (table, mem, _) = get_state(uc.get_data_mut());
            if let Some(dir) = table.asset_dirs.remove(&x0) {
                mem.free_heap(dir.name_buf);
            }
            mem.free_heap(x0);
            0
        }
        HleBuiltin::Adler32 => {
            // uLong adler32(uLong adler, const Bytef *buf, uInt len)
            if x1 == 0 {
                1
            } else {
                let len = (x2 as usize).min(64 * 1024 * 1024);
                let mut buf = vec![0u8; len];
                if uc.mem_read(x1, &mut buf).is_ok() {
                    zlib_adler32(x0 as u32, &buf) as u64
                } else {
                    x0
                }
            }
        }
        HleBuiltin::CompressBound => {
            // uLong compressBound(uLong sourceLen)
            x0.wrapping_add(x0 >> 12)
                .wrapping_add(x0 >> 14)
                .wrapping_add(x0 >> 25)
                .wrapping_add(13)
        }
        b @ (HleBuiltin::Compress | HleBuiltin::Compress2) => {
            // int compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)
            use std::io::Write;
            let level = if matches!(b, HleBuiltin::Compress2) {
                let lv = x4 as i32;
                if lv < 0 { 6 } else { (lv as u32).min(9) }
            } else {
                6
            };
            let dest_cap = read_guest_u64(uc, x1) as usize;
            let src_len = (x3 as usize).min(64 * 1024 * 1024);
            let mut src = vec![0u8; src_len];
            if x0 == 0 || x1 == 0 || (src_len > 0 && uc.mem_read(x2, &mut src).is_err()) {
                (-2i64) as u64 // Z_STREAM_ERROR
            } else {
                let mut enc =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
                if enc.write_all(&src).is_ok() {
                    if let Ok(out) = enc.finish() {
                        if out.len() > dest_cap {
                            (-5i64) as u64 // Z_BUF_ERROR
                        } else {
                            let _ = uc.mem_write(x0, &out);
                            let _ = uc.mem_write(x1, &(out.len() as u64).to_le_bytes());
                            0 // Z_OK
                        }
                    } else {
                        (-2i64) as u64
                    }
                } else {
                    (-2i64) as u64
                }
            }
        }
        HleBuiltin::Uncompress => {
            // int uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)
            use std::io::Read;
            let dest_cap = read_guest_u64(uc, x1) as usize;
            let src_len = (x3 as usize).min(64 * 1024 * 1024);
            let mut src = vec![0u8; src_len];
            if x0 == 0 || x1 == 0 || (src_len > 0 && uc.mem_read(x2, &mut src).is_err()) {
                (-2i64) as u64
            } else {
                let mut dec = flate2::read::ZlibDecoder::new(src.as_slice());
                let mut out = Vec::new();
                match dec.read_to_end(&mut out) {
                    Ok(_) => {
                        if out.len() > dest_cap {
                            (-5i64) as u64 // Z_BUF_ERROR
                        } else {
                            let _ = uc.mem_write(x0, &out);
                            let _ = uc.mem_write(x1, &(out.len() as u64).to_le_bytes());
                            0 // Z_OK
                        }
                    }
                    Err(_) => (-3i64) as u64, // Z_DATA_ERROR
                }
            }
        }
        b @ (HleBuiltin::DeflateInit | HleBuiltin::DeflateInit2) => {
            // deflateInit_(strm, level, version, stream_size)
            // deflateInit2_(strm, level, method, windowBits, memLevel, strategy, version, stream_size)
            if x0 == 0 {
                (-2i64) as u64
            } else {
                let lv_raw = x1 as i32;
                let level = if lv_raw < 0 { 6 } else { (lv_raw as u32).min(9) };
                let window_bits = if matches!(b, HleBuiltin::DeflateInit2) {
                    x3 as i32
                } else {
                    15
                };
                let zlib_header = window_bits >= 0;
                let _ = uc.mem_write(x0 + 0x10, &0u64.to_le_bytes()); // total_in
                let _ = uc.mem_write(x0 + 0x28, &0u64.to_le_bytes()); // total_out
                let _ = uc.mem_write(x0 + 0x30, &0u64.to_le_bytes()); // msg
                let _ = uc.mem_write(x0 + 0x38, &x0.to_le_bytes()); // state
                let (table, _, _) = get_state(uc.get_data_mut());
                table.zlib_streams.insert(
                    x0,
                    ZlibStreamState::Deflate {
                        compress: flate2::Compress::new(
                            flate2::Compression::new(level),
                            zlib_header,
                        ),
                        level,
                        zlib_header,
                    },
                );
                0
            }
        }
        HleBuiltin::Deflate => {
            // int deflate(z_streamp strm, int flush)
            let next_in = read_guest_u64(uc, x0);
            let mut a_in = [0u8; 4];
            let avail_in = if uc.mem_read(x0 + 0x08, &mut a_in).is_ok() {
                u32::from_le_bytes(a_in) as usize
            } else {
                0
            };
            let total_in = read_guest_u64(uc, x0 + 0x10);
            let next_out = read_guest_u64(uc, x0 + 0x18);
            let mut a_out = [0u8; 4];
            let avail_out = if uc.mem_read(x0 + 0x20, &mut a_out).is_ok() {
                u32::from_le_bytes(a_out) as usize
            } else {
                0
            };
            let total_out = read_guest_u64(uc, x0 + 0x28);
            let mut in_buf = vec![0u8; avail_in.min(16 * 1024 * 1024)];
            if !in_buf.is_empty() && next_in != 0 {
                let _ = uc.mem_read(next_in, &mut in_buf);
            }
            let mut out_buf = vec![0u8; avail_out.min(16 * 1024 * 1024)];
            let flush = match x1 as i32 {
                1 => flate2::FlushCompress::Partial,
                2 => flate2::FlushCompress::Sync,
                3 => flate2::FlushCompress::Full,
                4 => flate2::FlushCompress::Finish,
                _ => flate2::FlushCompress::None,
            };
            let res = {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(ZlibStreamState::Deflate { compress, .. }) =
                    table.zlib_streams.get_mut(&x0)
                {
                    let before_in = compress.total_in();
                    let before_out = compress.total_out();
                    match compress.compress(&in_buf, &mut out_buf, flush) {
                        Ok(status) => {
                            let consumed = (compress.total_in() - before_in) as usize;
                            let produced = (compress.total_out() - before_out) as usize;
                            let code = match status {
                                flate2::Status::Ok => 0i64,
                                flate2::Status::StreamEnd => 1i64,
                                flate2::Status::BufError => -5i64,
                            };
                            Some((consumed, produced, code))
                        }
                        Err(_) => Some((0, 0, -2i64)),
                    }
                } else {
                    None
                }
            };
            if let Some((consumed, produced, code)) = res {
                if produced > 0 && next_out != 0 {
                    let _ = uc.mem_write(next_out, &out_buf[..produced]);
                }
                let _ = uc.mem_write(x0, &(next_in + consumed as u64).to_le_bytes());
                let _ = uc.mem_write(
                    x0 + 0x08,
                    &(avail_in.saturating_sub(consumed) as u32).to_le_bytes(),
                );
                let _ = uc.mem_write(x0 + 0x10, &(total_in + consumed as u64).to_le_bytes());
                let _ = uc.mem_write(x0 + 0x18, &(next_out + produced as u64).to_le_bytes());
                let _ = uc.mem_write(
                    x0 + 0x20,
                    &(avail_out.saturating_sub(produced) as u32).to_le_bytes(),
                );
                let _ = uc.mem_write(x0 + 0x28, &(total_out + produced as u64).to_le_bytes());
                code as u64
            } else {
                (-2i64) as u64
            }
        }
        HleBuiltin::DeflateEnd => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.zlib_streams.remove(&x0);
            0
        }
        HleBuiltin::DeflateReset => {
            let (table, _, _) = get_state(uc.get_data_mut());
            if let Some(ZlibStreamState::Deflate {
                compress,
                level,
                zlib_header,
            }) = table.zlib_streams.get_mut(&x0)
            {
                *compress = flate2::Compress::new(flate2::Compression::new(*level), *zlib_header);
                let _ = uc.mem_write(x0 + 0x10, &0u64.to_le_bytes());
                let _ = uc.mem_write(x0 + 0x28, &0u64.to_le_bytes());
                0
            } else {
                (-2i64) as u64
            }
        }
        b @ (HleBuiltin::InflateInit | HleBuiltin::InflateInit2) => {
            // inflateInit_(strm, version, stream_size)
            // inflateInit2_(strm, windowBits, version, stream_size)
            if x0 == 0 {
                (-2i64) as u64
            } else {
                let window_bits = if matches!(b, HleBuiltin::InflateInit2) {
                    x1 as i32
                } else {
                    15
                };
                let zlib_header = window_bits > 0 && window_bits <= 15;
                let gzip_header_pending = window_bits >= 16;
                let _ = uc.mem_write(x0 + 0x10, &0u64.to_le_bytes());
                let _ = uc.mem_write(x0 + 0x28, &0u64.to_le_bytes());
                let _ = uc.mem_write(x0 + 0x30, &0u64.to_le_bytes());
                let _ = uc.mem_write(x0 + 0x38, &x0.to_le_bytes());
                let (table, _, _) = get_state(uc.get_data_mut());
                table.zlib_streams.insert(
                    x0,
                    ZlibStreamState::Inflate {
                        decompress: flate2::Decompress::new(zlib_header),
                        window_bits,
                        gzip_header_pending,
                    },
                );
                0
            }
        }
        HleBuiltin::Inflate => {
            // int inflate(z_streamp strm, int flush)
            let next_in = read_guest_u64(uc, x0);
            let mut a_in = [0u8; 4];
            let avail_in = if uc.mem_read(x0 + 0x08, &mut a_in).is_ok() {
                u32::from_le_bytes(a_in) as usize
            } else {
                0
            };
            let total_in = read_guest_u64(uc, x0 + 0x10);
            let next_out = read_guest_u64(uc, x0 + 0x18);
            let mut a_out = [0u8; 4];
            let avail_out = if uc.mem_read(x0 + 0x20, &mut a_out).is_ok() {
                u32::from_le_bytes(a_out) as usize
            } else {
                0
            };
            let total_out = read_guest_u64(uc, x0 + 0x28);
            let mut in_buf = vec![0u8; avail_in.min(16 * 1024 * 1024)];
            if !in_buf.is_empty() && next_in != 0 {
                let _ = uc.mem_read(next_in, &mut in_buf);
            }
            let mut out_buf = vec![0u8; avail_out.min(16 * 1024 * 1024)];
            let flush = match x1 as i32 {
                2 => flate2::FlushDecompress::Sync,
                4 => flate2::FlushDecompress::Finish,
                _ => flate2::FlushDecompress::None,
            };
            let res = {
                let (table, _, _) = get_state(uc.get_data_mut());
                if let Some(ZlibStreamState::Inflate {
                    decompress,
                    window_bits,
                    gzip_header_pending,
                }) = table.zlib_streams.get_mut(&x0)
                {
                    let mut header_skip = 0usize;
                    if *gzip_header_pending && !in_buf.is_empty() {
                        *gzip_header_pending = false;
                        if in_buf.len() >= 2 && in_buf[0] == 0x1F && in_buf[1] == 0x8B {
                            header_skip = parse_gzip_header_len(&in_buf);
                            *decompress = flate2::Decompress::new(false);
                        } else if *window_bits >= 32 {
                            *decompress = flate2::Decompress::new(true);
                        }
                    }
                    let slice = &in_buf[header_skip.min(in_buf.len())..];
                    let before_in = decompress.total_in();
                    let before_out = decompress.total_out();
                    match decompress.decompress(slice, &mut out_buf, flush) {
                        Ok(status) => {
                            let consumed = header_skip + (decompress.total_in() - before_in) as usize;
                            let produced = (decompress.total_out() - before_out) as usize;
                            let code = match status {
                                flate2::Status::Ok => 0i64,
                                flate2::Status::StreamEnd => 1i64,
                                flate2::Status::BufError => -5i64,
                            };
                            Some((consumed, produced, code))
                        }
                        Err(_) => Some((header_skip, 0, -3i64)),
                    }
                } else {
                    None
                }
            };
            if let Some((consumed, produced, code)) = res {
                if produced > 0 && next_out != 0 {
                    let _ = uc.mem_write(next_out, &out_buf[..produced]);
                }
                let _ = uc.mem_write(x0, &(next_in + consumed as u64).to_le_bytes());
                let _ = uc.mem_write(
                    x0 + 0x08,
                    &(avail_in.saturating_sub(consumed) as u32).to_le_bytes(),
                );
                let _ = uc.mem_write(x0 + 0x10, &(total_in + consumed as u64).to_le_bytes());
                let _ = uc.mem_write(x0 + 0x18, &(next_out + produced as u64).to_le_bytes());
                let _ = uc.mem_write(
                    x0 + 0x20,
                    &(avail_out.saturating_sub(produced) as u32).to_le_bytes(),
                );
                let _ = uc.mem_write(x0 + 0x28, &(total_out + produced as u64).to_le_bytes());
                code as u64
            } else {
                (-2i64) as u64
            }
        }
        HleBuiltin::InflateEnd => {
            let (table, _, _) = get_state(uc.get_data_mut());
            table.zlib_streams.remove(&x0);
            0
        }
        HleBuiltin::InflateReset => {
            let (table, _, _) = get_state(uc.get_data_mut());
            if let Some(ZlibStreamState::Inflate {
                decompress,
                window_bits,
                gzip_header_pending,
            }) = table.zlib_streams.get_mut(&x0)
            {
                let zlib_header = *window_bits > 0 && *window_bits <= 15;
                *decompress = flate2::Decompress::new(zlib_header);
                *gzip_header_pending = *window_bits >= 16;
                let _ = uc.mem_write(x0 + 0x10, &0u64.to_le_bytes());
                let _ = uc.mem_write(x0 + 0x28, &0u64.to_le_bytes());
                0
            } else {
                (-2i64) as u64
            }
        }
        HleBuiltin::UnimplementedStub(sym) => {
            let (table, _, vfs) = get_state(uc.get_data_mut());
            vfs.diag().record_missing_hle(&sym);
            let count = {
                let entry = table.unimplemented_hits.entry(sym.clone()).or_insert(0);
                *entry += 1;
                *entry
            };
            if count == 1 {
                vfs.breadcrumbs()
                    .push(format!("[HLE UNIMPL] {sym}(x0=0x{x0:x}, x1=0x{x1:x}) -> 0"));
                tracing::warn!(
                    target: "runidbg::hle",
                    symbol = %sym,
                    x0 = format_args!("0x{x0:x}"),
                    x1 = format_args!("0x{x1:x}"),
                    "调用未实现的外部符号（同符号后续命中自动聚合计数，可在 EnvDiag.missingHle 查看）"
                );
            }
            0
        }
    };

    let _ = uc.reg_write(RegisterARM64::X0, ret);
}

fn stdio_mode_to_flags(mode: &str) -> (u32, bool) {
    let m = mode.trim();
    if m.starts_with('w') {
        (crate::os::vfs::O_CREAT | crate::os::vfs::O_TRUNC, false)
    } else if m.starts_with('a') {
        (crate::os::vfs::O_CREAT, true)
    } else {
        (0, false)
    }
}

fn stdio_read_chunk(
    table: &mut TrampolineTable,
    vfs: &mut VirtualFs,
    fp: u64,
    max_len: usize,
) -> Option<Vec<u8>> {
    let st = table.stdio_streams.get_mut(&fp)?;
    if max_len == 0 {
        return Some(Vec::new());
    }
    let mut out = Vec::with_capacity(max_len.min(4096));
    if let Some(b) = st.ungetc_buf.take() {
        out.push(b);
    }
    let rem = max_len.saturating_sub(out.len());
    if rem > 0 {
        match vfs.read(st.fd, rem) {
            Ok(bytes) => {
                if bytes.len() < rem {
                    st.eof = true;
                }
                out.extend_from_slice(&bytes);
            }
            Err(_) => {
                st.err = true;
                return None;
            }
        }
    }
    Some(out)
}

fn stdio_seek_back(table: &mut TrampolineTable, vfs: &mut VirtualFs, fp: u64, back_bytes: usize) {
    if back_bytes == 0 {
        return;
    }
    if let Some(st) = table.stdio_streams.get_mut(&fp) {
        let _ = vfs.lseek(st.fd, -(back_bytes as i64), 1);
        st.eof = false;
    }
}

fn stdio_read_line(
    table: &mut TrampolineTable,
    vfs: &mut VirtualFs,
    fp: u64,
    max_bytes: usize,
    delim: u8,
) -> Option<Vec<u8>> {
    let chunk = stdio_read_chunk(table, vfs, fp, max_bytes)?;
    if chunk.is_empty() {
        return None;
    }
    if let Some(pos) = chunk.iter().position(|&b| b == delim) {
        let keep = pos + 1;
        let excess = chunk.len() - keep;
        if excess > 0 {
            stdio_seek_back(table, vfs, fp, excess);
        }
        Some(chunk[..keep].to_vec())
    } else {
        Some(chunk)
    }
}

/// ARM64 `setjmp`/`longjmp` 保存的 21 个 callee-saved 寄存器（共 168 字节，兼容 Bionic 256 字节 `jmp_buf`）
const JMPBUF_REGS: [RegisterARM64; 21] = [
    RegisterARM64::X19,
    RegisterARM64::X20,
    RegisterARM64::X21,
    RegisterARM64::X22,
    RegisterARM64::X23,
    RegisterARM64::X24,
    RegisterARM64::X25,
    RegisterARM64::X26,
    RegisterARM64::X27,
    RegisterARM64::X28,
    RegisterARM64::X29,
    RegisterARM64::X30,
    RegisterARM64::SP,
    RegisterARM64::D8,
    RegisterARM64::D9,
    RegisterARM64::D10,
    RegisterARM64::D11,
    RegisterARM64::D12,
    RegisterARM64::D13,
    RegisterARM64::D14,
    RegisterARM64::D15,
];

/// 标准 RFC 1950 `adler32` 校验和
fn zlib_adler32(adler: u32, data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut s1 = adler & 0xFFFF;
    let mut s2 = (adler >> 16) & 0xFFFF;
    for &b in data {
        s1 = (s1 + u32::from(b)) % MOD_ADLER;
        s2 = (s2 + s1) % MOD_ADLER;
    }
    (s2 << 16) | s1
}

/// 解析 RFC 1952 gzip 头部字节长度（用于 `inflateInit2_(strm, 16+15)` 流式解压）
fn parse_gzip_header_len(buf: &[u8]) -> usize {
    if buf.len() < 10 || buf[0] != 0x1F || buf[1] != 0x8B {
        return 0;
    }
    let flg = buf[3];
    let mut pos = 10;
    // FEXTRA
    if (flg & 0x04) != 0 && pos + 2 <= buf.len() {
        let xlen = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
        pos = (pos + 2 + xlen).min(buf.len());
    }
    // FNAME
    if (flg & 0x08) != 0 {
        while pos < buf.len() {
            let b = buf[pos];
            pos += 1;
            if b == 0 {
                break;
            }
        }
    }
    // FCOMMENT
    if (flg & 0x10) != 0 {
        while pos < buf.len() {
            let b = buf[pos];
            pos += 1;
            if b == 0 {
                break;
            }
        }
    }
    // FHCRC
    if (flg & 0x02) != 0 && pos + 2 <= buf.len() {
        pos += 2;
    }
    pos
}
