# pto.set_flag

`pto.set_flag` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Signal an event from one pipeline to another pipeline.

## Mechanism

`pto.set_flag` marks the named event as produced by the source pipeline and visible to the destination pipeline. It is the producer half of the explicit event protocol used to connect MTE, vector, and other execution stages.

## Syntax

### PTO Assembly Form

```text
set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

### AS Level 1 (SSA)

```mlir
pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| `SRC_PIPE` | pipe identifier | Pipeline that produces the event |
| `DST_PIPE` | pipe identifier | Pipeline that is allowed to consume the event |
| `EVENT_ID` | event identifier | Named event slot used for the producer-consumer edge |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it updates pipeline-event state. |

## Side Effects

Signals the named event, making later `pto.wait_flag` operations on the matching `(SRC_PIPE, DST_PIPE, EVENT_ID)` tuple eligible to unblock.

## Constraints

- The selected pipe identifiers and event identifier MUST be valid for the selected target profile.
- The event protocol is directional: the producer and consumer pipe roles matter.
- Portable code MUST pair event signaling with the corresponding wait in the intended dependency chain.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- CPU simulation preserves the visible event protocol but may not expose all low-level hazards that motivate it on hardware.
- A2/A3 and A5 may use different concrete pipe and event spaces; portable code must rely on the documented PTO contract plus the selected target profile.

## Examples

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: (none)
- Next op in instruction set: [pto.wait_flag](./wait-flag.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
