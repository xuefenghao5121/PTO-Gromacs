#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TaxpyTest, UpdatesDestinationInPlace)
{
    using TileData = Tile<TileType::Vec, float, 2, 8, BLayout::RowMajor, 2, 4>;

    TileData dst;
    TileData src;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src);

    const float dstValues[2][4] = {{1.0f, 2.0f, 3.0f, 4.0f}, {-1.0f, -2.0f, -3.0f, -4.0f}};
    const float srcValues[2][4] = {{0.5f, 1.0f, 1.5f, 2.0f}, {2.0f, 1.5f, 1.0f, 0.5f}};
    constexpr float scalar = -2.0f;

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            SetValue(dst, r, c, dstValues[r][c]);
            SetValue(src, r, c, srcValues[r][c]);
        }
    }

    TAXPY(dst, src, scalar);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const float expected = dstValues[r][c] + srcValues[r][c] * scalar;
            ExpectValueEquals(GetValue(dst, r, c), expected);
        }
    }
}

} // namespace
