#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TRowProdTest, ProducesProductPerRow)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 1>;
    using TmpTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;

    SrcTile src;
    DstTile dst;
    TmpTile tmp;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst, tmp);

    const int values[3][4] = {{2, 3, 4, 5}, {1, -2, 3, -4}, {7, 0, 2, 9}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TROWPROD(dst, src, tmp);

    const int expected[3] = {120, 24, 0};
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        ExpectValueEquals(GetValue(dst, r, 0), expected[r]);
    }
}

} // namespace
