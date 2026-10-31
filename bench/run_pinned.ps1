# Builds and runs every benchmark pinned to one performance core, so that a
# hybrid CPU (P- and E-cores) does not move it mid-run:
#   bench_goose_<config>.csv, bench_numpy.csv, bench_rust_<config>.csv, bench_cpp.csv
#
#   powershell -File bench\run_pinned.ps1 [-Core 4] [-Only goose|numpy|rust|cpp]
#
# Run it on a quiet machine: other load moves the numbers by more than the
# differences being measured.
param([int]$Core = 4, [string]$Only = "")

$here = $PSScriptRoot
$goose = if ($env:GOOSE) { $env:GOOSE } else { "goose" }
$one = [long](1 -shl $Core)

function Run-Pinned($exe, $argList, $log, $dir = $here) {
    $p = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $dir -NoNewWindow -PassThru `
        -RedirectStandardOutput $log
    $p.ProcessorAffinity = [IntPtr]$one
    $p.WaitForExit()
}

# A failed build must not leave a stale executable to be measured.
function Build($src, $exe, $flags) {
    Remove-Item $exe -ErrorAction SilentlyContinue
    $env:CL = $flags
    & $goose build -o $exe $src | Out-Null
    Remove-Item Env:CL -ErrorAction SilentlyContinue
    if (-not (Test-Path $exe)) { throw "building $src failed" }
}

if ($Only -eq "" -or $Only -eq "goose") {
    $configs = [ordered]@{ "base" = ""; "avx2" = "/arch:AVX2" }
    foreach ($name in $configs.Keys) {
        Build "$here\bench.goose" "$here\bench_$name.exe" $configs[$name]
        Run-Pinned "$here\bench_$name.exe" @("bench_goose_$name.csv") "$here\log_goose_$name.txt"
        Write-Output "goose $name done"
    }
}
if ($Only -eq "" -or $Only -eq "numpy") {
    Run-Pinned "python" @("bench_numpy.py", "bench_numpy.csv") "$here\log_numpy.txt"
    Write-Output "numpy done"
}
if ($Only -eq "" -or $Only -eq "rust") {
    foreach ($cfg in @("base", "native")) {
        $env:RUSTFLAGS = if ($cfg -eq "native") { "-C target-cpu=native" } else { "" }
        $env:CARGO_TARGET_DIR = "$here\rust\target_$cfg"
        cargo build --release --manifest-path "$here\rust\Cargo.toml" 2>&1 | Out-Null
        Remove-Item Env:RUSTFLAGS, Env:CARGO_TARGET_DIR
        $exe = "$here\rust\target_$cfg\release\bench_rand.exe"
        if (-not (Test-Path $exe)) { throw "building the Rust benchmark failed" }
        Run-Pinned $exe @("..\bench_rust_$cfg.csv") "$here\log_rust_$cfg.txt" "$here\rust"
        Write-Output "rust $cfg done"
    }
}
if ($Only -eq "" -or $Only -eq "cpp") {
    Push-Location $here
    Remove-Item bench_cpp.exe -ErrorAction SilentlyContinue
    clang-cl /nologo /O2 /arch:AVX2 /EHsc bench_cpp.cpp /Fe:bench_cpp.exe | Out-Null
    Pop-Location
    if (-not (Test-Path "$here\bench_cpp.exe")) { throw "building bench_cpp.cpp failed" }
    Run-Pinned "$here\bench_cpp.exe" @("bench_cpp.csv") "$here\log_cpp.txt"
    Write-Output "c++ done"
}
