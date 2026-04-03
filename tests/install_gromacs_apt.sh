#!/bin/bash
# GROMACS 快速安装脚本 - 使用apt包管理器
# 需要 sudo 权限

set -e

echo "======================================"
echo "GROMACS 快速安装 (Ubuntu 24.04)"
echo "======================================"
echo ""

echo "[1/3] 更新软件源..."
sudo apt-get update

echo ""
echo "[2/3] 安装GROMACS..."
sudo apt-get install -y gromacs

echo ""
echo "[3/3] 验证安装..."
gmx --version

echo ""
echo "======================================"
echo "安装成功!"
echo "======================================"
echo ""
echo "GROMACS 已安装到系统目录"
echo "可以直接使用 'gmx' 命令"
