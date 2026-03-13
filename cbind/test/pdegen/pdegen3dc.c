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
  // err = psb_c_dvect_reinit(bh, false);
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

  // Now I will test that I can change the values again

  start_ptr = psb_c_dvect_f_get_pnt(bh);
  end_ptr = start_ptr + nl;
  while (start_ptr < end_ptr) {
    *start_ptr = 69.; // val[0];
    start_ptr++;
  }

  // verify the entries of the vector are correct
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
    fprintf(stdout, "Process %d vector entries check OK\n", iam);

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