#!/bin/bash
# GROMACS 安装脚本 - 用户目录安装
# 适用于: Ubuntu 24.04, Intel i7-13700 (AVX2)

set -e

INSTALL_DIR="${HOME}/.local/gromacs"
BUILD_DIR="/tmp/gromacs-build"
GROMACS_VERSION="2024.1"
JOBS=$(nproc)

echo "======================================"
echo "GROMACS ${GROMACS_VERSION} 安装脚本"
echo "======================================"
echo "安装目录: ${INSTALL_DIR}"
echo "编译核心: ${JOBS}"
echo ""

# 检查依赖
check_dependencies() {
    echo "[1/5] 检查依赖..."
    local missing=()
    
    for cmd in cmake make gcc g++ wget; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo "缺少依赖: ${missing[*]}"
        echo "请运行: sudo apt-get install -y cmake build-essential wget git"
        exit 1
    fi
    
    echo "依赖检查通过"
}

# 下载GROMACS
download_gromacs() {
    echo "[2/5] 下载GROMACS ${GROMACS_VERSION}..."
    
    if [ -d "${BUILD_DIR}/gromacs-${GROMACS_VERSION}" ]; then
        echo "源码已存在，跳过下载"
        return
    fi
    
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    
    wget -q --show-progress \
        "https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz" \
        -O gromacs.tar.gz
    
    tar xzf gromacs.tar.gz
    rm gromacs.tar.gz
    
    echo "下载完成"
}

# 配置编译选项
configure_gromacs() {
    echo "[3/5] 配置编译选项..."
    
    cd ${BUILD_DIR}/gromacs-${GROMACS_VERSION}
    mkdir -p build
    cd build
    
    cmake .. \
        -DGMX_INSTALL_PREFIX=${INSTALL_DIR} \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DGMX_GPU=OFF \
        -DGMX_MPI=OFF \
        -DGMX_SIMD=AVX2_256 \
        -DGMX_OPENMP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGMX_BUILD_UNITTESTS=ON \
        -DGMX_BUILD_BENCHMARKS=ON
    
    echo "配置完成"
}

# 编译安装
build_install() {
    echo "[4/5] 编译GROMACS (使用 ${JOBS} 核心)..."
    
    cd ${BUILD_DIR}/gromacs-${GROMACS_VERSION}/build
    
    make -j${JOBS}
    
    echo "[5/5] 安装..."
    make install
    
    echo "安装完成"
}

# 配置环境
setup_environment() {
    echo ""
    echo "======================================"
    echo "配置环境变量"
    echo "======================================"
    
    echo ""
    echo "请将以下内容添加到 ~/.bashrc:"
    echo ""
    echo "# GROMACS"
    echo "source ${INSTALL_DIR}/bin/GMXRC"
    echo ""
    echo "或者临时使用:"
    echo "source ${INSTALL_DIR}/bin/GMXRC"
}

# 验证安装
verify_installation() {
    echo ""
    echo "验证安装..."
    
    source ${INSTALL_DIR}/bin/GMXRC
    gmx --version
    
    echo ""
    echo "======================================"
    echo "安装成功!"
    echo "======================================"
}

# 主流程
main() {
    check_dependencies
    download_gromacs
    configure_gromacs
    build_install
    setup_environment
    verify_installation
}

main "$@"
