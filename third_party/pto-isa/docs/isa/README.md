<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Manual And Reference

This directory is the canonical PTO ISA tree. It combines the architecture manual, the instruction set guides, the instruction set contracts, and the exact instruction-reference groupings in one place.

## Textual Assembly Inside PTO ISA

This tree is the canonical PTO ISA manual. Textual assembly spelling belongs to the PTO ISA syntax instruction set, not to a second parallel architecture manual.

- PTO ISA defines architecture-visible semantics, legality, state, ordering, target-profile boundaries, and the visible behavior of `pto.t*`, `pto.v*`, `pto.*`, and other operations.
- PTO-AS is the assembler-facing spelling used to write those operations and operands. It is part of how PTO ISA is expressed, not a separate ISA with different semantics.

If the question is "what does this legal PTO program mean across CPU, A2/A3, and A5?", stay in this tree. If the question is "what is the operand shape or textual spelling of this operation?", use the syntax-and-operands pages in this same tree.

## Start Here

## Axis Reduce / Expand
- [TROWSUM](TROWSUM.md) - Reduce each row by summing across columns.
- [TROWPROD](TROWPROD.md) - Reduce each row by multiplying across columns.
- [TCOLSUM](TCOLSUM.md) - Reduce each column by summing across rows.
- [TCOLPROD](TCOLPROD.md) - Reduce each column by multiplying across rows.
- [TCOLMAX](TCOLMAX.md) - Reduce each column by taking the maximum across rows.
- [TROWMAX](TROWMAX.md) - Reduce each row by taking the maximum across columns.
- [TROWMIN](TROWMIN.md) - Reduce each row by taking the minimum across columns.
- [TROWARGMAX](TROWARGMAX.md) - Get the column index of the maximum element for each row.
- [TROWARGMIN](TROWARGMIN.md) - Get the column index of the minimum element for each row.
- [TCOLARGMAX](TCOLARGMAX.md) - Get the row index of the maximum element for each column.
- [TCOLARGMIN](TCOLARGMIN.md) - Get the row index of the minimum element for each column.
- [TROWEXPAND](TROWEXPAND.md) - Broadcast the first element of each source row across the destination row.
- [TROWEXPANDDIV](TROWEXPANDDIV.md) - Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`.
- [TROWEXPANDMUL](TROWEXPANDMUL.md) - Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.
- [TROWEXPANDSUB](TROWEXPANDSUB.md) - Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`.
- [TROWEXPANDADD](TROWEXPANDADD.md) - Row-wise broadcast add: add a per-row scalar vector.
- [TROWEXPANDMAX](TROWEXPANDMAX.md) - Row-wise broadcast max with a per-row scalar vector.
- [TROWEXPANDMIN](TROWEXPANDMIN.md) - Row-wise broadcast min with a per-row scalar vector.
- [TROWEXPANDEXPDIF](TROWEXPANDEXPDIF.md) - Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars.
- [TCOLMIN](TCOLMIN.md) - Reduce each column by taking the minimum across rows.
- [TCOLEXPAND](TCOLEXPAND.md) - Broadcast the first element of each source column across the destination column.
- [TCOLEXPANDDIV](TCOLEXPANDDIV.md) - Column-wise broadcast divide: divide each column by a per-column scalar vector.
- [TCOLEXPANDMUL](TCOLEXPANDMUL.md) - Column-wise broadcast multiply: multiply each column by a per-column scalar vector.
- [TCOLEXPANDADD](TCOLEXPANDADD.md) - Column-wise broadcast add with per-column scalar vector.
- [TCOLEXPANDMAX](TCOLEXPANDMAX.md) - Column-wise broadcast max with per-column scalar vector.
- [TCOLEXPANDMIN](TCOLEXPANDMIN.md) - Column-wise broadcast min with per-column scalar vector.
- [TCOLEXPANDSUB](TCOLEXPANDSUB.md) - Column-wise broadcast subtract: subtract a per-column scalar vector from each column.
- [TCOLEXPANDEXPDIF](TCOLEXPANDEXPDIF.md) - Column-wise exp-diff: compute exp(src0 - src1) with per-column scalars.

## Model Layers

Reading order matches the manual chapter map: programming and machine models, then syntax and state, then memory, then opcode reference.

- [Programming model](programming-model/tiles-and-valid-regions.md)
- [Machine model](machine-model/execution-agents.md)
- [Syntax and operands](syntax-and-operands/assembly-model.md)
- [Type system](state-and-types/type-system.md)
- [Location intent and legality](state-and-types/location-intent-and-legality.md)
- [Memory model](memory-model/consistency-baseline.md)

## Complex
- [TPRINT](TPRINT.md) - Debug/print elements from a tile (implementation-defined).
- [TMRGSORT](TMRGSORT.md) - Merge sort for multiple sorted lists (implementation-defined element format and layout).
- [TSORT32](TSORT32.md) - Sort each 32-element block of `src` together with the corresponding indices from `idx`, and write the sorted value-index pairs into `dst`.
- [TGATHER](TGATHER.md) - Gather/select elements using either an index tile or a compile-time mask pattern.
- [TCI](TCI.md) - Generate a contiguous integer sequence into a destination tile.
- [TTRI](TTRI.md) - Generate a triangular (lower/upper) mask tile.
- [TRANDOM](TRANDOM.md) - Generates random numbers in the destination tile using a counter-based cipher algorithm.
- [TPARTADD](TPARTADD.md) - Partial elementwise add with implementation-defined handling of mismatched valid regions.
- [TPARTMUL](TPARTMUL.md) - Partial elementwise multiply with implementation-defined handling of mismatched valid regions.
- [TPARTMAX](TPARTMAX.md) - Partial elementwise max with implementation-defined handling of mismatched valid regions.
- [TPARTMIN](TPARTMIN.md) - Partial elementwise min with implementation-defined handling of mismatched valid regions.
- [TGATHERB](TGATHERB.md) - Gather elements using byte offsets.
- [TSCATTER](TSCATTER.md) - Scatter rows of a source tile into a destination tile using per-element row indices.
- [TQUANT](TQUANT.md) - Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs.

- [Instruction overview](instruction-surfaces/README.md)
- [Instruction set contracts](instruction-families/README.md)
- [Format of instruction descriptions](reference/format-of-instruction-descriptions.md)
- [Tile instruction reference](tile/README.md)
- [Vector instruction reference](vector/README.md)
- [Scalar and control reference](scalar/README.md)
- [Other and communication reference](other/README.md)
- [Common conventions](conventions.md)

## Supporting Reference

- [Reference notes](reference/README.md) (glossary, diagnostics, portability, source of truth)

## Compatibility Wrappers

The grouped instruction set trees under `tile/`, `vector/`, `scalar/`, and `other/` are the canonical PTO ISA paths.

Some older root-level tile pages such as `TADD.md`, `TLOAD.md`, and `TMATMUL.md` now remain only as compatibility wrappers so existing links do not break immediately. New PTO ISA documentation should link to the grouped instruction set paths, especially the standalone per-op pages under:

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
