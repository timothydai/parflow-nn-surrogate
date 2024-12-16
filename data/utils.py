import glob
import json
import os

import numpy as np
import pandas as pd
from parflowio import pyParflowio
import torch


def collect_targets_from_one_simulation(simulation_dir, num_ts_to_sample=None):
    pfb_files = sorted(simulation_dir.glob("*.press.*.pfb"))
    if len(pfb_files) == 0:
        raise ValueError(f"No .pfb files found in {str(simulation_dir)}.")
    elif num_ts_to_sample and len(pfb_files) < num_ts_to_sample:
        raise ValueError(
            f"Not enough *.press.*.pfb files to sample {num_ts_to_sample} time steps. "
            f"Got {len(pfb_files)} press.*.pfb files."
        )
    ts = None
    if num_ts_to_sample:
        ts = _sample_ts_geometrically(len(pfb_files) - 1, num_ts_to_sample)
    else:
        ts = range(1, len(pfb_files))

    out = []
    for t in ts:
        pfb_file = pfb_files[t]
        y_t = _pf_read(str(pfb_file))
        y_t = torch.from_numpy(y_t)
        out.append(y_t)

    out = torch.stack(out, dim=0)  # List[(x, y, z)] --> [t, x, y, z].
    out = out.unsqueeze(-1)  # [t, x, y, z] --> [t, x, y, z, c].

    initial_pressure = _pf_read(str(pfb_files[0]))
    initial_pressure = torch.from_numpy(initial_pressure)
    return out, initial_pressure, ts


def collect_static_inputs_from_one_simulation(simulation_dir, initial_pressure):
    pfmetadata_files = list(simulation_dir.glob("*.pfmetadata"))
    permx_files = list(simulation_dir.glob("*.perm_x.pfb"))
    permz_files = list(simulation_dir.glob("*.perm_z.pfb"))
    porosity_files = list(simulation_dir.glob("*.porosity.pfb"))
    alpha_files = list(simulation_dir.glob("*.alpha.pfb"))
    n_files = list(simulation_dir.glob("*.n.pfb"))
    sres_files = list(simulation_dir.glob("*.sres.pfb"))

    assert len(pfmetadata_files) == 1
    assert len(permx_files) == 1
    assert len(permz_files) == 1
    assert len(porosity_files) == 1
    assert len(alpha_files) == 1
    assert len(n_files) == 1
    assert len(sres_files) == 1

    pfmetadata = _parse_pfmetadata(pfmetadata_files[0])
    permx = torch.from_numpy(_pf_read(str(permx_files[0])))
    permz = torch.from_numpy(_pf_read(str(permz_files[0])))
    poros = torch.from_numpy(_pf_read(str(porosity_files[0])))
    alpha = torch.from_numpy(_pf_read(str(alpha_files[0])))
    n = torch.from_numpy(_pf_read(str(n_files[0])))
    sres = torch.from_numpy(_pf_read(str(sres_files[0])))
    cell_volume = torch.from_numpy(_get_cell_volume_arr(pfmetadata))

    static_inputs = torch.stack(
        [
            permx,
            permz,
            poros,
            alpha,
            n,
            sres,
            cell_volume,
            initial_pressure,
        ],
        axis=-1,
    )  # List[x, y, z] --> [x, y, z, c].
    return static_inputs


def collect_dynamic_inputs_from_one_simulation(simulation_dir, ts, mode):
    nldas_files = list(simulation_dir.glob("clm_input/nldas.dat"))
    assert len(nldas_files) == 1

    nldas_file = nldas_files[0]
    dynamic_inputs = _sample_nldas(nldas_file, ts)
    dynamic_inputs = torch.from_numpy(dynamic_inputs)  # [t, c].

    if mode == "stage3":
        tcl_scripts = sorted((simulation_dir / "tcl_scripts").glob("*.tcl"))
        # We only supply the initial BC pressure.
        bc_pressure = _tcl_path_to_bcpressure(tcl_scripts[0])
        bc_pressure = torch.ones([dynamic_inputs.shape[0], 1]) * bc_pressure
        dynamic_inputs = torch.concat([dynamic_inputs, bc_pressure], axis=-1)

    return dynamic_inputs  # [t, c].


def _sample_nldas(nldas_path, ts, hours_per_t=24):
    nldas = []
    with open(nldas_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        nldas.append(list(map(float, line.split())))
    nldas = np.array(nldas)

    nldas_sampled = []
    ts = [0] + ts
    for i in range(len(ts) - 1):
        sample = nldas[
            ts[i] * hours_per_t : ts[i + 1] * hours_per_t
        ]  # .shape[0] is number of hours.
        sample = sample.mean(0, keepdims=True)
        nldas_sampled.append(sample)
    nldas_sampled = np.concatenate(nldas_sampled, 0)  # List[1, c] --> [t, c].
    return nldas_sampled.copy()


def _tcl_path_to_bcpressure(tcl_path):
    with open(tcl_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Patch.infiltration.BCPressure.alltime.Value" in line:
            return float(line.split()[-1])


def _sample_ts_geometrically(total_ts, target_num_ts):
    # Sn = a (1 - r ** n) / (1 - r) ==> ar**n - Snr + Sn - a = 0
    coefficients = [0 for _ in range(target_num_ts + 1)]
    coefficients[0] = 1
    coefficients[-2] = -total_ts
    coefficients[-1] = total_ts - 1

    r = max(
        list(
            map(
                lambda x: x.real,
                list(filter(lambda x: x.imag == 0, np.roots(coefficients).tolist())),
            )
        )
    )

    t = 0
    dt = 1
    ts = []
    for i in range(target_num_ts):
        t += dt * r**i
        ts.append(int(t))

    return ts


def _pf_read(pfb_file_path):
    pfb_file = pyParflowio.PFData(pfb_file_path)
    pfb_file.loadHeader()
    pfb_file.loadData()
    arr = pfb_file.moveDataArray()
    pfb_file.close()
    assert arr.ndim == 3, "Only 3D arrays are supported"
    # NOTE: parflowio seems to read arrays such that the rows (i.e. axis=1) are
    # reversed w.r.t what pfio gives us. Hence the np.flip.
    arr = np.flip(arr, axis=1)
    arr = np.transpose(arr, (2, 1, 0))  # [z, y, x] --> [x, y, z].
    return arr.copy()  # To prevent negative strides.


def _parse_pfmetadata(pfmetadata_path):
    with open(pfmetadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def _get_cell_volume_arr(run_metadata):
    Nx = int(run_metadata["inputs"]["configuration"]["data"]["ComputationalGrid.NX"])
    Ny = int(run_metadata["inputs"]["configuration"]["data"]["ComputationalGrid.NY"])
    Nz = int(run_metadata["inputs"]["configuration"]["data"]["ComputationalGrid.NZ"])
    dx = float(run_metadata["inputs"]["configuration"]["data"]["ComputationalGrid.DX"])
    dy = float(run_metadata["inputs"]["configuration"]["data"]["ComputationalGrid.DY"])
    dz = float(run_metadata["inputs"]["configuration"]["data"]["ComputationalGrid.DZ"])
    assert (
        run_metadata["inputs"]["configuration"]["data"]["Solver.Nonlinear.VariableDz"]
        == "True"
    )
    if (
        run_metadata["inputs"]["configuration"]["data"]["Solver.Nonlinear.VariableDz"]
        == "True"
    ):
        x = np.ones((Nz, Ny, Nx)) * dx
        y = np.ones((Nz, Ny, Nx)) * dy
        z = np.ones((Nz, Ny, Nx)) * dz
        dz_scales = [
            float(
                run_metadata["inputs"]["configuration"]["data"][
                    f"Cell.{i}.dzScale.Value"
                ]
            )
            for i in range(
                int(
                    run_metadata["inputs"]["configuration"]["data"][
                        "dzScale.nzListNumber"
                    ]
                )
            )
        ]
        for i in range(len(dz_scales)):
            z[i, ...] *= dz_scales[i]
    out = x * y * z
    out = np.transpose(out, (2, 1, 0))  # [z, y, x] --> [x, y, z].
    return out.copy()
