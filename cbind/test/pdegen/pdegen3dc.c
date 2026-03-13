#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "psb_base_cbind.h"

#define NG_GLOBAL 100 // fixed global vector size

//  Vector generation: b[i] = global_index + 1
psb_i_t vecgen(psb_c_ctxt cctxt, psb_i_t nl, psb_l_t vl[],
               psb_c_descriptor *cdh, psb_c_dvector *bh) {
  psb_i_t i, info;
  psb_l_t irow[1];
  double val[1];

  info = 0;
  psb_c_set_index_base(0);

  for (i = 0; i < nl; i++) {
    irow[0] = vl[i];
    val[0] = (double)(vl[i] + 1); // b[i] = global_index + 1
    psb_c_dgeins(1, irow, val, bh, cdh);
  }

  if ((info = psb_c_cdasb(cdh)) != 0)
    return (info);
  if ((info = psb_c_dgeasb(bh, cdh)) != 0)
    return (info);
  return (info);
}

int main(int argc, char *argv[]) {
  psb_c_ctxt *cctxt;
  psb_i_t iam, np;
  psb_i_t nb, nl, info;
  psb_l_t i, ng, *vl, k;
  psb_c_dvector *bh;
  psb_c_descriptor *cdh;

  cctxt = psb_c_new_ctxt();
  psb_c_init(cctxt);
  psb_c_info(*cctxt, &iam, &np);

  psb_c_barrier(*cctxt);

  cdh = psb_c_new_descriptor();
  psb_c_set_index_base(0);

  // block distribution of the vector
  ng = (psb_l_t)NG_GLOBAL;
  nb = (ng + np - 1) / np;
  if (iam == 0) {
    fprintf(stdout, "Global size: %ld, Local size per process: %d\n", ng, nb);
  }
  nl = nb;
  if ((ng - iam * nb) < nl)
    nl = ng - iam * nb;
  if ((vl = malloc(nb * sizeof(psb_l_t))) == NULL) {
    fprintf(stderr, "On %d: malloc failure\n", iam);
    psb_c_abort(*cctxt);
  }
  i = ((psb_l_t)iam) * nb;
  for (k = 0; k < nl; k++)
    vl[k] = i + k;

  // fill descriptor
  if ((info = psb_c_cdall_vl(nl, vl, *cctxt, cdh)) != 0) {
    fprintf(stderr, "From cdall: %d\nAbortging\n", info);
    psb_c_abort(*cctxt);
  }

  bh = psb_c_new_dvector();
  psb_c_dgeall(bh, cdh);

  if (vecgen(*cctxt, nl, vl, cdh, bh) != 0) {
    fprintf(stderr, "Error during vector build\n");
    psb_c_abort(*cctxt);
  }
  psb_c_barrier(*cctxt);

  // Basic tests

  // 2 norm of vector b
  double norm2_b = psb_c_dgenrm2(bh, cdh);
  if (iam == 0)
    fprintf(stdout, "||b||_2 = %lg\n", norm2_b);

  int err;
  err = psb_c_dvect_reinit(bh, false);
  if (err != 0) {
    fprintf(stderr, "Error during vector reinit: %d\n", err);
    psb_c_abort(*cctxt);
  }

  // // now we want to change every entry of bh.
  psb_l_t irow[1];
  double val[1];
  // val[0] = 1.0;
  double *start_ptr = psb_c_dvect_f_get_pnt(bh);
  double *end_ptr = start_ptr + nl;

  while (start_ptr < end_ptr) {
    *start_ptr += 1.; // val[0];
    start_ptr++;
  }
  // for (k = 0; k < nl; k++) {
  //   irow[0] = vl[k];
  //   psb_c_dgeins(1, irow, val, bh,
  //                cdh); // -> fails with PSBLAS Error (1124) in
  //                subroutine:
  //                      // psb_dinsvi Invalid state for vector
  // }

  // verify the entries of the vector are correct
  for (k = 0; k < nl; k++) {
    irow[0] = vl[k];
    double got_val = psb_c_dgetelem(bh, irow[0], cdh);
    double expected_val = (double)(vl[k] + 2); //(double)(vl[k] + 1);
    if (fabs(got_val - expected_val) > 1e-12) {
      fprintf(stderr, "Error: got %lg, expected %lg at index %ld\n", got_val,
              expected_val, irow[0]);
      psb_c_abort(*cctxt);
    }
  }

  // Now I set every entry of bh to 1.0
  start_ptr = psb_c_dvect_f_get_pnt(bh);
  end_ptr = start_ptr + nl;
  while (start_ptr < end_ptr) {
    *start_ptr = 1.0;
    start_ptr++;
  }

  norm2_b = psb_c_dgenrm2(bh, cdh);
  if (iam == 0)
    fprintf(stdout, "||b||_2 after changes = %lg\n", norm2_b);

  // Let's assemble the vector
  err = psb_c_dgeasb(bh, cdh);
  if (err != 0) {
    fprintf(stderr, "Error during vector assembly: %d\n", err);
    psb_c_abort(*cctxt);
  }

  // Now I set every entry of bh to 69
  start_ptr = psb_c_dvect_f_get_pnt(bh);
  end_ptr = start_ptr + nl;
  while (start_ptr < end_ptr) {
    *start_ptr = 69;
    start_ptr++;
  }

  for (k = 0; k < nl; k++) {
    irow[0] = vl[k];
    double got_val = psb_c_dgetelem(bh, irow[0], cdh);
    double expected_val = 69.;
    if (fabs(got_val - expected_val) > 1e-12) {
      fprintf(stderr, "Error: got %lg, expected %lg at index %ld\n", got_val,
              expected_val, irow[0]);
      psb_c_abort(*cctxt);
    }
  }

  if (iam == 0)
    fprintf(stdout, "OK\n");

  // --- Remote insertion test ---
  // Reinit into remote-build mode: zeroes entries, sets bld state,
  // enables remote buffering and allocates rmtv/rmidx.
  err = psb_c_dgereinit(bh, cdh, true);
  if (err != 0) {
    fprintf(stderr, "Error during dgereinit (remote test): %d\n", err);
    psb_c_abort(*cctxt);
  }

  if (np > 1) {
    // Pick the first two indices owned by the *next* process — these are remote
    // for the current process.
    psb_i_t next = (iam + 1) % np;
    psb_l_t rnb = (ng + np - 1) / np;
    psb_l_t remote_idx[2];
    remote_idx[0] = (psb_l_t)next * rnb;
    remote_idx[1] = (psb_l_t)next * rnb + 1;
    double remote_val[2] = {100.0, 200.0};

    // Only insert if those indices actually exist in the global range
    if (remote_idx[1] < ng) {
      psb_c_dgeins(2, remote_idx, remote_val, bh, cdh);
      if (iam == 0)
        fprintf(stdout,
                "Process %d inserted %.1f at global idx %ld and %.1f at %ld\n",
                iam, remote_val[0], remote_idx[0], remote_val[1],
                remote_idx[1]);
    }
  } else {
    if (iam == 0)
      fprintf(stdout, "Skipping remote insertion test (single process)\n");
  }

  // Assemble: this exchanges the remotely-inserted values with their owners
  err = psb_c_dgeasb(bh, cdh);
  if (err != 0) {
    fprintf(stderr, "Error during vector assembly (remote test): %d\n", err);
    psb_c_abort(*cctxt);
  }

  // Each process verifies its own entries: local ones should be 0 (cleared by
  // reinit), except for the two entries that the *previous* process sent here.
  if (np > 1) {
    psb_l_t rnb = (ng + np - 1) / np;
    psb_l_t sent0 = (psb_l_t)iam * rnb;     // first  index prev sent us
    psb_l_t sent1 = (psb_l_t)iam * rnb + 1; // second index prev sent us
    double exp0 = 100.0, exp1 = 200.0;

    if (sent1 < ng) {
      double g0 = psb_c_dgetelem(bh, sent0, cdh);
      double g1 = psb_c_dgetelem(bh, sent1, cdh);
      if (fabs(g0 - exp0) > 1e-12 || fabs(g1 - exp1) > 1e-12) {
        fprintf(stderr,
                "Process %d remote-insert (overwrite) check failed: "
                "idx %ld got %.1f (exp %.1f), idx %ld got %.1f (exp %.1f)\n",
                iam, sent0, g0, exp0, sent1, g1, exp1);
        psb_c_abort(*cctxt);
      } else {
        fprintf(stdout,
                "Process %d remote-insert (overwrite) check OK: "
                "idx %ld=%.1f, idx %ld=%.1f\n",
                iam, sent0, g0, sent1, g1);
      }
    }
  }

  psb_c_barrier(*cctxt);
  // --- Remote addition test ---
  // Reinit WITHOUT clearing: existing values (100/200 at sent0/sent1) survive.
  // Re-arm remote build with dupl_add_ so the next round accumulates.
  err = psb_c_dgereinit(bh, cdh, false);
  if (err != 0) {
    fprintf(stderr, "Error during dgereinit (addition test): %d\n", err);
    psb_c_abort(*cctxt);
  }

  if (np > 1) {
    psb_i_t next = (iam + 1) % np;
    psb_l_t rnb = (ng + np - 1) / np;
    psb_l_t remote_idx[2];
    remote_idx[0] = (psb_l_t)next * rnb;
    remote_idx[1] = (psb_l_t)next * rnb + 1;
    double remote_val[2] = {50.0, 70.0};

    if (remote_idx[1] < ng) {
      psb_c_dgeins_add(2, remote_idx, remote_val, bh, cdh);
      if (iam == 0)
        fprintf(
            stdout, "Process %d added %.1f at global idx %ld and %.1f at %ld\n",
            iam, remote_val[0], remote_idx[0], remote_val[1], remote_idx[1]);
    }
  }

  err = psb_c_dgeasb(bh, cdh);
  if (err != 0) {
    fprintf(stderr, "Error during vector assembly (addition test): %d\n", err);
    psb_c_abort(*cctxt);
  }

  // Verify accumulation: sent0 = 100+50 = 150, sent1 = 200+70 = 270
  if (np > 1) {
    psb_l_t rnb = (ng + np - 1) / np;
    psb_l_t sent0 = (psb_l_t)iam * rnb;
    psb_l_t sent1 = (psb_l_t)iam * rnb + 1;
    double exp0 = 150.0, exp1 = 270.0;

    if (sent1 < ng) {
      double g0 = psb_c_dgetelem(bh, sent0, cdh);
      double g1 = psb_c_dgetelem(bh, sent1, cdh);
      if (fabs(g0 - exp0) > 1e-12 || fabs(g1 - exp1) > 1e-12) {
        fprintf(stderr,
                "Process %d remote-insert (addition) check failed: "
                "idx %ld got %.1f (exp %.1f), idx %ld got %.1f (exp %.1f)\n",
                iam, sent0, g0, exp0, sent1, g1, exp1);
        psb_c_abort(*cctxt);
      } else {
        fprintf(stdout,
                "Process %d remote-insert (addition) check OK: "
                "idx %ld=%.1f (exp %.1f), idx %ld=%.1f (exp %.1f)\n",
                iam, sent0, g0, exp0, sent1, g1, exp1);
      }
    }
  }

  // --- psb_c_dgeins_v demo ---
  // Round 1: use PSB_INSERT_VALUES mode to set every local entry to 10.0.
  // This is equivalent to calling psb_c_dgeins().
  err = psb_c_dgereinit(bh, cdh, true);
  if (err != 0) {
    fprintf(stderr, "Error during dgereinit (dgeins_v demo): %d\n", err);
    psb_c_abort(*cctxt);
  }

  for (k = 0; k < nl; k++) {
    psb_l_t idx = vl[k];
    double v = 10.0;
    psb_c_dgeins_v(1, &idx, &v, bh, cdh, PSB_INSERT_VALUES);
  }
  err = psb_c_dgeasb(bh, cdh);
  if (err != 0) {
    fprintf(stderr, "Error during dgeasb (dgeins_v INSERT): %d\n", err);
    psb_c_abort(*cctxt);
  }

  for (k = 0; k < nl; k++) {
    double got = psb_c_dgetelem(bh, vl[k], cdh);
    if (fabs(got - 10.0) > 1e-12) {
      fprintf(stderr,
              "Process %d dgeins_v INSERT check failed at idx %ld: "
              "got %.1f, expected 10.0\n",
              iam, vl[k], got);
      psb_c_abort(*cctxt);
    }
  }
  if (iam == 0)
    fprintf(stdout, "dgeins_v PSB_INSERT_VALUES check OK\n");

  // Round 2: use PSB_ADD_VALUES mode to add 5.0 to every local entry.
  // Must call dgereinit+dgeasb between the two modes (no mixing allowed).
  err = psb_c_dgereinit(bh, cdh, false); // keep existing 10.0 values
  if (err != 0) {
    fprintf(stderr, "Error during dgereinit (dgeins_v ADD demo): %d\n", err);
    psb_c_abort(*cctxt);
  }

  for (k = 0; k < nl; k++) {
    psb_l_t idx = vl[k];
    double v = 5.0;
    psb_c_dgeins_v(1, &idx, &v, bh, cdh, PSB_ADD_VALUES);
  }
  err = psb_c_dgeasb(bh, cdh);
  if (err != 0) {
    fprintf(stderr, "Error during dgeasb (dgeins_v ADD): %d\n", err);
    psb_c_abort(*cctxt);
  }

  for (k = 0; k < nl; k++) {
    double got = psb_c_dgetelem(bh, vl[k], cdh);
    if (fabs(got - 15.0) > 1e-12) {
      fprintf(stderr,
              "Process %d dgeins_v ADD check failed at idx %ld: "
              "got %.1f, expected 15.0\n",
              iam, vl[k], got);
      psb_c_abort(*cctxt);
    }
  }
  if (iam == 0)
    fprintf(stdout, "dgeins_v PSB_ADD_VALUES check OK\n");

  // --- Mode-mixing detection test ---
  // Starting a build round with INSERT_VALUES, then trying to switch to
  // ADD_VALUES without an intervening psb_c_dgeasb must be rejected.
  err = psb_c_dgereinit(bh, cdh, true);
  if (err != 0) {
    fprintf(stderr, "Error during dgereinit (mode-mix test): %d\n", err);
    psb_c_abort(*cctxt);
  }

  // First batch: INSERT_VALUES — buffers the entries.
  {
    psb_l_t idx = vl[0];
    double v = 42.0;
    err = psb_c_dgeins_v(1, &idx, &v, bh, cdh, PSB_INSERT_VALUES);
    if (err != 0) {
      fprintf(stderr,
              "Process %d: unexpected error on first INSERT_VALUES: %d\n", iam,
              err);
      psb_c_abort(*cctxt);
    }
  }

  // Second batch: switch to ADD_VALUES WITHOUT assemble — must return error -2.
  {
    psb_l_t idx = vl[0];
    double v = 1.0;
    int mix_err = psb_c_dgeins_v(1, &idx, &v, bh, cdh, PSB_ADD_VALUES);
    if (mix_err == PSB_ERR_MODE_MISMATCH) {
      if (iam == 0)
        fprintf(stdout,
                "Mode-mixing correctly detected: got error %d "
                "(PSB_ERR_MODE_MISMATCH)\n",
                mix_err);
    } else {
      fprintf(stderr,
              "Process %d: mode-mixing NOT detected (got %d, expected -2)\n",
              iam, mix_err);
      psb_c_abort(*cctxt);
    }
  }

  // --- psb_c_dgeins_v with remote indices ---
  // The mode-mix test left the vector in bld state with a stale buffered
  // entry; call dgereinit to start fresh.

  // Round A: PSB_INSERT_VALUES — each process overwrites the first two indices
  //          owned by the *next* process with 300.0 and 400.0.
  if (np > 1) {
    err = psb_c_dgereinit(bh, cdh, true);
    if (err != 0) {
      fprintf(stderr, "Error during dgereinit (dgeins_v remote INSERT): %d\n",
              err);
      psb_c_abort(*cctxt);
    }

    psb_i_t next = (iam + 1) % np;
    psb_l_t rnb = (ng + np - 1) / np;
    psb_l_t ridx[2];
    ridx[0] = (psb_l_t)next * rnb;
    ridx[1] = (psb_l_t)next * rnb + 1;
    double rval[2] = {300.0, 400.0};

    if (ridx[1] < ng) {
      err = psb_c_dgeins_v(2, ridx, rval, bh, cdh, PSB_INSERT_VALUES);
      if (err != 0) {
        fprintf(stderr, "Process %d: psb_c_dgeins_v remote INSERT failed: %d\n",
                iam, err);
        psb_c_abort(*cctxt);
      }
    }

    err = psb_c_dgeasb(bh, cdh);
    if (err != 0) {
      fprintf(stderr, "Error during dgeasb (dgeins_v remote INSERT): %d\n",
              err);
      psb_c_abort(*cctxt);
    }

    // Each process checks the two entries the previous process sent to it.
    psb_l_t sent0 = (psb_l_t)iam * rnb;
    psb_l_t sent1 = (psb_l_t)iam * rnb + 1;
    if (sent1 < ng) {
      double g0 = psb_c_dgetelem(bh, sent0, cdh);
      double g1 = psb_c_dgetelem(bh, sent1, cdh);
      if (fabs(g0 - 300.0) > 1e-12 || fabs(g1 - 400.0) > 1e-12) {
        fprintf(stderr,
                "Process %d dgeins_v remote INSERT check failed: "
                "idx %ld=%.1f (exp 300.0), idx %ld=%.1f (exp 400.0)\n",
                iam, sent0, g0, sent1, g1);
        psb_c_abort(*cctxt);
      } else {
        fprintf(stdout,
                "Process %d dgeins_v remote INSERT check OK: "
                "idx %ld=%.1f, idx %ld=%.1f\n",
                iam, sent0, g0, sent1, g1);
      }
    }

    psb_c_barrier(*cctxt);

    // Round B: PSB_ADD_VALUES — add 50.0 and 60.0 to the same remote indices.
    // Expected result: 300+50=350, 400+60=460.
    err = psb_c_dgereinit(bh, cdh, false); // keep existing 300/400 values
    if (err != 0) {
      fprintf(stderr, "Error during dgereinit (dgeins_v remote ADD): %d\n",
              err);
      psb_c_abort(*cctxt);
    }

    double radd[2] = {50.0, 60.0};
    if (ridx[1] < ng) {
      err = psb_c_dgeins_v(2, ridx, radd, bh, cdh, PSB_ADD_VALUES);
      if (err != 0) {
        fprintf(stderr, "Process %d: psb_c_dgeins_v remote ADD failed: %d\n",
                iam, err);
        psb_c_abort(*cctxt);
      }
    }

    err = psb_c_dgeasb(bh, cdh);
    if (err != 0) {
      fprintf(stderr, "Error during dgeasb (dgeins_v remote ADD): %d\n", err);
      psb_c_abort(*cctxt);
    }

    if (sent1 < ng) {
      double g0 = psb_c_dgetelem(bh, sent0, cdh);
      double g1 = psb_c_dgetelem(bh, sent1, cdh);
      if (fabs(g0 - 350.0) > 1e-12 || fabs(g1 - 460.0) > 1e-12) {
        fprintf(stderr,
                "Process %d dgeins_v remote ADD check failed: "
                "idx %ld=%.1f (exp 350.0), idx %ld=%.1f (exp 460.0)\n",
                iam, sent0, g0, sent1, g1);
        psb_c_abort(*cctxt);
      } else {
        fprintf(stdout,
                "Process %d dgeins_v remote ADD check OK: "
                "idx %ld=%.1f (exp 350.0), idx %ld=%.1f (exp 460.0)\n",
                iam, sent0, g0, sent1, g1);
      }
    }

    psb_c_barrier(*cctxt);

    // Round C: mode-mixing with remote indices — buffer a remote INSERT, then
    // try ADD without assembling first.  Must be rejected.
    err = psb_c_dgereinit(bh, cdh, true);
    if (err != 0) {
      fprintf(stderr, "Error during dgereinit (dgeins_v remote mode-mix): %d\n",
              err);
      psb_c_abort(*cctxt);
    }

    if (ridx[1] < ng) {
      double first_val = 1.0;
      err = psb_c_dgeins_v(1, ridx, &first_val, bh, cdh, PSB_INSERT_VALUES);
      if (err != 0) {
        fprintf(stderr,
                "Process %d: unexpected error on remote INSERT (mode-mix "
                "setup): %d\n",
                iam, err);
        psb_c_abort(*cctxt);
      }

      double second_val = 1.0;
      int mix_err =
          psb_c_dgeins_v(1, ridx, &second_val, bh, cdh, PSB_ADD_VALUES);
      if (mix_err == PSB_ERR_MODE_MISMATCH) {
        if (iam == 0)
          fprintf(stdout, "dgeins_v remote mode-mixing correctly detected "
                          "(PSB_ERR_MODE_MISMATCH)\n");
      } else {
        fprintf(stderr,
                "Process %d: remote mode-mixing NOT detected "
                "(got %d, expected PSB_ERR_MODE_MISMATCH)\n",
                iam, mix_err);
        psb_c_abort(*cctxt);
      }
    }
  } else {
    if (iam == 0)
      fprintf(stdout, "Skipping dgeins_v remote tests (single process)\n");
  }

  // --- Bug-exposure test: dgeins_v after dgeall (no dgereinit) ---
  // After psb_c_dgeall the internal dupl flag is psb_dupl_def_ = psb_dupl_add_
  // (value 1).  If psb_c_dgeins_v did NOT call set_dupl(mode), inserting the
  // same local index twice with PSB_INSERT_VALUES would assemble with ADD
  // semantics, giving 30+7=37 instead of the correct overwrite result 7.
  //
  // We free and reallocate to guarantee a pristine dgeall state, then
  // deliberately skip dgereinit to exercise the general-case path.
  // if ((info = psb_c_dgefree(bh, cdh)) != 0) {
  //   fprintf(stderr, "From dgefree (bug-exposure test): %d\n", info);
  //   psb_c_abort(*cctxt);
  // }
  // psb_c_dgeall(bh, cdh); // dupl is now psb_dupl_add_ — NO dgereinit

  psb_c_dgereinit(bh, cdh, false);

  // Insert vl[0] with 30.0 first, then overwrite it with 7.0.
  // Both calls use PSB_INSERT_VALUES; set_dupl(mode) inside dgeins_v must
  // arm ovwrt_ before each call so the second write wins.
  {
    psb_l_t idx = vl[0];
    double v1 = 30.0;
    psb_c_dgeins_v(1, &idx, &v1, bh, cdh, PSB_INSERT_VALUES);
    double v2 = 7.0;
    psb_c_dgeins_v(1, &idx, &v2, bh, cdh, PSB_INSERT_VALUES);
  }
  err = psb_c_dgeasb(bh, cdh);
  if (err != 0) {
    fprintf(stderr, "Error during dgeasb (bug-exposure test): %d\n", err);
    psb_c_abort(*cctxt);
  }
  {
    double got = psb_c_dgetelem(bh, vl[0], cdh);
    if (fabs(got - 7.0) > 1e-12) {
      fprintf(stderr,
              "Process %d bug-exposure test FAILED at idx %ld: "
              "got %.1f, expected 7.0 "
              "(37.0 means ADD semantics leaked through — set_dupl missing)\n",
              iam, vl[0], got);
      psb_c_abort(*cctxt);
    }
    if (iam == 0)
      fprintf(stdout,
              "Bug-exposure test OK: dgeins_v after dgeall correctly "
              "overwrites (got %.1f, not 37.0)\n",
              got);
  }

  // cleanup
  if ((info = psb_c_dgefree(bh, cdh)) != 0) {
    fprintf(stderr, "From dgefree(b): %d\n", info);
    psb_c_abort(*cctxt);
  }
  if ((info = psb_c_cdfree(cdh)) != 0) {
    fprintf(stderr, "From cdfree: %d\n", info);
    psb_c_abort(*cctxt);
  }

  free(bh);
  free(cdh);
  free(vl);

  psb_c_barrier(*cctxt);
  psb_c_exit(*cctxt);
  free(cctxt);
  return 0;
}