set -e
echo "=== 清理 CUDA JIT 缓存 ==="
rm -rf ~/.nv/ComputeCache ~/.nv/GLCache
rm -rf ~/.cache/nvidia

echo "=== 清理 SAPIEN / OptiX 着色器缓存 ==="
rm -rf ~/.cache/sapien
rm -rf ~/.cache/OptiX* 2>/dev/null || true
rm -rf /tmp/optix_cache* /tmp/sapien_* 2>/dev/null || true

echo "=== 清理 Vulkan shader 缓存 ==="
rm -rf ~/.cache/vulkan
rm -rf ~/.cache/mesa_shader_cache 2>/dev/null || true

echo "=== 清理 OIDN kernel 缓存 ==="
rm -rf ~/.cache/OpenImageDenoise 2>/dev/null || true
rm -rf /tmp/oidn_* 2>/dev/null || true