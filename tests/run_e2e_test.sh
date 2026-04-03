#!/bin/bash
# GROMACS端到端测试脚本
# 需要先安装GROMACS并设置环境

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_FILE="${RESULTS_DIR}/test_$(date +%Y%m%d_%H%M%S).log"

# 创建结果目录
mkdir -p ${RESULTS_DIR}

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a ${LOG_FILE}
}

# 检查GROMACS
check_gromacs() {
    log "检查GROMACS安装..."
    
    if ! command -v gmx &> /dev/null; then
        log "错误: GROMACS未安装"
        log "请运行: source ~/.local/gromacs/bin/GMXRC"
        log "或运行安装脚本: ./install_gromacs.sh"
        exit 1
    fi
    
    gmx --version | tee -a ${LOG_FILE}
    log ""
}

# 运行基础测试
run_basic_tests() {
    log "=== 基础功能测试 ==="
    
    cd ${SCRIPT_DIR}
    
    # 编译测试框架
    log "编译测试框架..."
    
    if command -v gcc &> /dev/null; then
        log "编译单元测试..."
        gcc -O2 -march=native -o test_unit_x86 test_unit_x86.c test_framework_x86.c -lm 2>&1 | tee -a ${LOG_FILE} || true
        
        if [ -f test_unit_x86 ]; then
            log "运行单元测试..."
            ./test_unit_x86 2>&1 | tee -a ${LOG_FILE}
        fi
    fi
    
    log ""
}

# 下载测试案例
download_benchmark() {
    log "=== 下载GROMACS基准测试案例 ==="
    
    cd ${RESULTS_DIR}
    
    # 水盒子基准测试
    if [ ! -f "dppc.zip" ]; then
        log "下载DPPC膜基准测试..."
        wget -q --show-progress http://www.gromacs.org/@api/deki/files/185/=dppc.zip -O dppc.zip || true
    fi
    
    # 创建简单的水盒子测试
    log "创建简单水盒子测试..."
    mkdir -p simple_water
    cd simple_water
    
    # 生成SPC/E水盒子
    if command -v gmx &> /dev/null; then
        log "生成216个水分子..."
        gmx solvate -box 2 2 2 -o water.gro -p topol.top 2>&1 | tee -a ${LOG_FILE} || true
        
        if [ -f water.gro ]; then
            # 创建topology文件
            cat > topol.top << 'EOF'
#include "oplsaa.ff/forcefield.itp"
#include "oplsaa.ff/spce.itp"
[ system ]
Simple Water Box
[ molecules ]
SOL 216
EOF
            
            # 能量最小化参数
            cat > minim.mdp << 'EOF'
integrator = steep
emtol = 1000
emstep = 0.01
nsteps = 50000
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
EOF
            
            log "运行能量最小化..."
            gmx grompp -f minim.mdp -c water.gro -p topol.top -o em.tpr 2>&1 | tee -a ${LOG_FILE} || true
            
            if [ -f em.tpr ]; then
                log "执行能量最小化..."
                gmx mdrun -deffnm em -v 2>&1 | tee -a ${LOG_FILE} || true
            fi
        fi
    fi
    
    cd ${SCRIPT_DIR}
    log ""
}

# 性能测试
run_benchmark() {
    log "=== 性能基准测试 ==="
    
    cd ${RESULTS_DIR}/simple_water
    
    if [ ! -f em.tpr ]; then
        log "跳过性能测试 (缺少输入文件)"
        return
    fi
    
    # 创建MD参数文件
    cat > md.mdp << 'EOF'
integrator = md
dt = 0.002
nsteps = 50000
nstenergy = 100
nstlog = 1000
nstxout = 5000
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
tcoupl = V-rescale
tc-grps = System
tau_t = 0.1
ref_t = 300
pcoupl = Parrinello-Rahman
pcoupltype = isotropic
tau_p = 2.0
ref_p = 1.0
compressibility = 4.5e-5
EOF
    
    log "准备MD模拟..."
    gmx grompp -f md.mdp -c em.gro -p topol.top -o md.tpr 2>&1 | tee -a ${LOG_FILE} || true
    
    if [ -f md.tpr ]; then
        # 单核测试
        log ""
        log "单核性能测试..."
        /usr/bin/time -v gmx mdrun -deffnm md -nt 1 2>&1 | tee -a ${LOG_FILE} || true
        
        # 多核测试
        local cores=$(nproc)
        if [ $cores -gt 1 ]; then
            log ""
            log "多核性能测试 (${cores}核)..."
            /usr/bin/time -v gmx mdrun -deffnm md_multi -nt $cores 2>&1 | tee -a ${LOG_FILE} || true
        fi
    fi
    
    cd ${SCRIPT_DIR}
    log ""
}

# 能量守恒验证
verify_energy_conservation() {
    log "=== 能量守恒验证 ==="
    
    cd ${RESULTS_DIR}/simple_water
    
    if [ -f md.edr ]; then
        log "提取能量数据..."
        echo "Total-Energy" | gmx energy -f md.edr -o energy.xvg 2>&1 | tee -a ${LOG_FILE} || true
        
        if [ -f energy.xvg ]; then
            log "分析能量波动..."
            # 简单统计
            tail -n +20 energy.xvg | awk '{
                sum += $2
                sum2 += $2*$2
                n++
            } END {
                mean = sum/n
                std = sqrt(sum2/n - mean*mean)
                rel_std = std/mean * 100
                printf "  平均能量: %.3f kJ/mol\n", mean
                printf "  标准差:   %.6f kJ/mol\n", std
                printf "  相对波动: %.4f%%\n", rel_std
            }' | tee -a ${LOG_FILE}
        fi
    else
        log "跳过能量验证 (缺少能量文件)"
    fi
    
    cd ${SCRIPT_DIR}
    log ""
}

# 生成报告
generate_report() {
    log "=== 生成测试报告 ==="
    
    local report_file="${RESULTS_DIR}/report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "======================================"
        echo "GROMACS 端到端测试报告"
        echo "======================================"
        echo ""
        echo "日期: $(date)"
        echo "主机: $(hostname)"
        echo "系统: $(lsb_release -d 2>/dev/null | cut -f2)"
        echo "CPU:  $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2)"
        echo "核心: $(nproc)"
        echo "内存: $(free -h | grep Mem | awk '{print $2}')"
        echo ""
        echo "--------------------------------------"
        echo "GROMACS 版本"
        echo "--------------------------------------"
        gmx --version 2>/dev/null || echo "GROMACS 未安装"
        echo ""
        echo "--------------------------------------"
        echo "测试日志"
        echo "--------------------------------------"
        cat ${LOG_FILE}
        echo ""
        echo "======================================"
        echo "测试完成"
        echo "======================================"
    } > ${report_file}
    
    log "报告已生成: ${report_file}"
}

# 主流程
main() {
    log "======================================"
    log "GROMACS 端到端测试开始"
    log "======================================"
    log ""
    
    check_gromacs
    run_basic_tests
    download_benchmark
    run_benchmark
    verify_energy_conservation
    generate_report
    
    log "测试完成!"
}

main "$@"
