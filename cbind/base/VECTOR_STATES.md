# PSBLAS Dense Vector States — C Binding Guide

## Overview

A `psb_c_dvector` (Fortran: `psb_d_vect_type`) is always in one of four internal
states. The state controls which operations are legal and how insertion calls
behave. Mixing operations that belong to different states leads to silent
corruption or runtime errors.

---

## States

### `null_`

Set automatically on fresh allocation before any `geall` call.  
No data, no operations possible. Transition out via `psb_c_dgeall`.

---

### `bld_` — Build state

**Set by:** `psb_c_dgeall` or `psb_c_dgereinit(clear=true)`

**Internal behaviour:**  
Insertion calls (`dgeins`, `dgeins_add`, `dgeins_v`) do **not** write directly
to the assembled data array `x%v`. Instead they **append** `(index, value)`
pairs onto a COO staging buffer (`x%iv` / `x%v`), incrementing `ncfs` (number
of coefficients staged). The `dupl` flag is **irrelevant during insertion** — it
is only applied at assembly time when `dgeasb` scatters the COO buffer into the
final dense array.

**Allowed operations:**
- `psb_c_dgeins` / `psb_c_dgeins_add` / `psb_c_dgeins_v` — buffer values
- `psb_c_dgeasb` — scatter COO buffer → `x%v` using `dupl` semantics, then
  transition to `asb_`

**Forbidden:**
- Reading values meaningfully (`dgetelem` will return the unscattered buffer)
- Mixing `INSERT_VALUES` and `ADD_VALUES` calls inside the same build round
  without an intervening `dgeasb`

**Note on remote indices:**  
Indices not owned by the local process are stored in a separate remote buffer
(`x%rmtv` / `x%rmidx`, size tracked by `nrmv`). They are exchanged with their
owners during `dgeasb`.

---

### `upd_` — Update state

**Set by:** `psb_c_dgereinit(clear=false)` or `psb_c_dvect_reinit`

**Internal behaviour:**  
Insertion calls write **directly** to `x%v(local_idx)`, applying `dupl`
semantics immediately (no staging, no COO buffer). `ncfs` is always 0 in this
state.

**Allowed operations:**
- `psb_c_dgeins` / `psb_c_dgeins_add` / `psb_c_dgeins_v` — direct scatter
- `psb_c_dvect_f_get_pnt` + pointer arithmetic — direct memory access
- `psb_c_dgeasb` — mostly a no-op (reallocates if needed), transitions to `asb_`

---

### `asb_` — Assembled state

**Set by:** `psb_c_dgeasb`

This is the **only state** in which the vector is safe for use in computations.

**Allowed operations:**
- All math: `dgenrm2`, dot products, axpby, etc.
- `psb_c_dgetelem` — read individual elements
- `psb_c_dvect_f_get_pnt` — direct pointer access
- `psb_c_dvect_reinit` → transitions to `upd_`
- `psb_c_dgereinit(clear=true)` → transitions to `bld_`
- `psb_c_dgereinit(clear=false)` → transitions to `upd_`

**Forbidden:**
- `psb_c_dgeins` — will error with "Invalid state for vector" (PSBLAS error 1124)

---

## State Transition Diagram

```
           psb_c_dgeall
null_ ─────────────────────► bld_
                               │  ▲
                    dgeasb     │  │  dgereinit(clear=true)
                               ▼  │
                              asb_ ◄──────────────── upd_
                               │                      ▲
                               │  dgereinit(false)    │ dgeasb
                               └──────────────────► upd_
                                   dvect_reinit
```

---

## `dupl` semantics summary

| Constant           | Value | Applied in `bld_` | Applied in `upd_` |
|--------------------|-------|-------------------|-------------------|
| `psb_dupl_add_`    |   1   | at `dgeasb`       | immediately       |
| `psb_dupl_ovwrt_`  |   2   | at `dgeasb`       | immediately       |

C header aliases (`psb_c_base.h`):

```c
#define PSB_INSERT_VALUES  psb_dupl_ovwrt_   /* = 2 */
#define PSB_ADD_VALUES     psb_dupl_add_     /* = 1 */
```

---

## Key invariants

1. **Always call `dgereinit` before a batch of `dgeins_v` calls.**  
   Without it the vector is in `bld_` state with `dupl = psb_dupl_add_` (the
   default after `dgeall`), so `INSERT_VALUES` inserts will assemble with ADD
   semantics, silently corrupting results.

2. **Always call `dgeasb` before reading values or performing math.**

3. **Never mix `INSERT_VALUES` and `ADD_VALUES` in the same build round.**  
   `psb_c_dgeins_v` enforces this: if a buffered entry already exists
   (`ncfs > 0` or `nrmv > 0`) and the requested mode differs from the current
   `dupl`, it returns `PSB_ERR_MODE_MISMATCH` (`-2`).
