    import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import argparse
import os
import numpy as np
from pathlib import Path
import json
import time
import pandas as pd
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw

# needed to load extensions
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

# VIZ
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sortingview.views as vv

# AIND
from aind_data_schema.core.processing import DataProcess


URL = "https://github.com/AllenNeuralDynamics/aind-ephys-visualization"
VERSION = "1.0"

GH_CURATION_REPO = "gh://AllenNeuralDynamics/ephys-sorting-manual-curation/main"
LABEL_CHOICES = ["noise", "MUA", "SUA", "pMUA", "pSUA"]

data_folder = Path("../data/")
scratch_folder = Path("../scratch")
results_folder = Path("../results/")

# Define argument parser
parser = argparse.ArgumentParser(description="Curate ecephys data")

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="-1", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    args = parser.parse_args()

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    N_JOBS = int(N_JOBS_CO) if N_JOBS_CO is not None else N_JOBS

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_visualization"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    visualization_params = processing_params["visualization"]

    ###### VISUALIZATION #########
    print("\n\nVISUALIZATION")
    t_visualization_start_all = time.perf_counter()
    datetime_start_visualization = datetime.now()

    # check if test
    if (data_folder / "postprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_folder = data_folder / "postprocessing_pipeline_output_test"
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
        curation_folder = data_folder / "curation_pipeline_output_test"
        unit_classifier_folder = data_folder / "unit_classifier_pipeline_output_test"
        spikesorted_folder = data_folder / "spikesorting_pipeline_output_test"
        skip_timeseries = False
    else:
        postprocessed_folder = data_folder
        preprocessed_folder = data_folder
        curation_folder = data_folder
        unit_classifier_folder = data_folder
        spikesorted_folder = data_folder
        data_processes_spikesorting_folder = data_folder
        skip_timeseries = False

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session_folder = ecephys_sessions[0]

    # in pipeline the ephys folder is renames 'ecephys_session'
    # in this case, grab session name from data_description (if it exists)
    data_description_file = session_folder / "data_description.json"
    if data_description_file.is_file():
        with open(data_description_file, "r") as f:
            data_description_dict = json.load(f)
        session_name = data_description_dict["name"]
    else:
        session_name = session_folder.name

    print(f"Session name: {session_name}")

    # Retrieve recording_names from preprocessed folder
    recording_names = [
        "_".join(p.name.split("_")[1:])
        for p in preprocessed_folder.iterdir()
        if p.is_dir() and "preprocessed_" in p.name
    ]

    # loop through block-streams
    for recording_name in recording_names:
        t_visualization_start = time.perf_counter()
        datetime_start_visualization = datetime.now()
        visualization_output = {}

        recording_folder = preprocessed_folder / f"preprocessed_{recording_name}"
        analyzer_folder = postprocessed_folder / f"postprocessed_{recording_name}"
        preprocessed_json_file = preprocessed_folder / f"preprocessedviz_{recording_name}.json"
        qc_file = curation_folder / f"qc_{recording_name}.npy"
        unit_classifier_file = unit_classifier_folder / f"unit_classifier_{recording_name}.csv"
        motion_folder = preprocessed_folder / f"motion_{recording_name}"
        visualization_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        # save vizualization output
        visualization_output_file = results_folder / f"visualization_{recording_name}.json"

        print(f"Visualizing recording: {recording_name}")

        with open(preprocessed_json_file, "r") as f:
            preprocessing_vizualization_data = json.load(f)

        # drift
        cmap = plt.get_cmap(visualization_params["drift"]["cmap"])
        norm = Normalize(
            vmin=visualization_params["drift"]["vmin"], vmax=visualization_params["drift"]["vmax"], clip=True
        )
        n_skip = visualization_params["drift"]["n_skip"]
        alpha = visualization_params["drift"]["alpha"]

        # use spike locations
        skip_drift = False
        spike_locations_available = False
        # use spike locations
        if analyzer_folder.is_dir():
            try:
                analyzer = si.load_sorting_analyzer(analyzer_folder)
                # here recording_folder MUST exist
                assert recording_folder.is_dir(), f"Recording folder {recording_folder} does not exist"
                recording = si.load_extractor(recording_folder)
                if analyzer.has_extension("spike_locations"):
                    print(f"\tVisualizing drift maps using spike sorted data")
                    peaks = analyzer.sorting.to_spike_vector()
                    peak_locations = analyzer.get_extension("spike_locations").get_data()
                    peak_amps = analyzer.get_extension("spike_amplitudes").get_data()
                    spike_locations_available = True
            except:
                print(
                    f"\tCould not load sorting analyzer or recording for {recording_name}"
                )

        # if spike locations are not available, detect and localize peaks
        if not spike_locations_available:
            from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
            from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
            from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass

            print(f"\tVisualizing drift maps using detected peaks (no spike sorting available)")
            # locally_exclusive + pipeline steps LocalizeCenterOfMass + PeakToPeakFeature
            drift_data = preprocessing_vizualization_data[recording_name]["drift"]
            try:
                recording = si.load_extractor(drift_data["recording"], base_folder=preprocessed_folder)

                # Here we use the node pipeline implementation
                peak_detector_node = DetectPeakLocallyExclusive(recording, **visualization_params["drift"]["detection"])
                extract_dense_waveforms_node = ExtractDenseWaveforms(
                    recording,
                    ms_before=visualization_params["drift"]["localization"]["ms_before"],
                    ms_after=visualization_params["drift"]["localization"]["ms_after"],
                    parents=[peak_detector_node],
                    return_output=False,
                )
                localize_peaks_node = LocalizeCenterOfMass(
                    recording,
                    radius_um=visualization_params["drift"]["localization"]["radius_um"],
                    parents=[peak_detector_node, extract_dense_waveforms_node],
                )
                pipeline_nodes = [peak_detector_node, extract_dense_waveforms_node, localize_peaks_node]
                peaks, peak_locations = run_node_pipeline(
                    recording, nodes=pipeline_nodes, job_kwargs=si.get_global_job_kwargs()
                )
                print(f"\tDetected {len(peaks)} peaks")
                peak_amps = peaks["amplitude"]
            except Exception as e:
                print(f"\t\tCould not load drift recording. Error:\n{e}\nSkipping")
                skip_drift = True

        if not skip_drift:
            y_locs = recording.get_channel_locations()[:, 1]
            ylim = [np.min(y_locs), np.max(y_locs)]

            fig_drift, axs_drift = plt.subplots(
                ncols=recording.get_num_segments(), figsize=visualization_params["drift"]["figsize"]
            )
            for segment_index in range(recording.get_num_segments()):
                segment_mask = peaks["segment_index"] == segment_index
                x = peaks[segment_mask]["sample_index"] / recording.sampling_frequency
                y = peak_locations[segment_mask]["y"]
                # subsample
                x_sub = x[::n_skip]
                y_sub = y[::n_skip]
                a_sub = peak_amps[::n_skip]
                colors = cmap(norm(a_sub))

                if recording.get_num_segments() == 1:
                    ax_drift = axs_drift
                else:
                    ax_drift = axs_drift[segment_index]
                ax_drift.scatter(x_sub, y_sub, s=1, c=colors, alpha=alpha)
                ax_drift.set_xlabel("time (s)", fontsize=12)
                ax_drift.set_ylabel("depth ($\\mu$m)", fontsize=12)
                ax_drift.set_xlim(
                    0, recording.get_num_samples(segment_index=segment_index) / recording.sampling_frequency
                )
                ax_drift.set_ylim(ylim)
                ax_drift.spines["top"].set_visible(False)
                ax_drift.spines["right"].set_visible(False)
            fig_drift_folder = scratch_folder / "drift_maps"
            fig_drift_folder.mkdir(exist_ok=True)
            fig_drift.savefig(fig_drift_folder / f"{recording_name}_drift.png", dpi=300)

            if not skip_timeseries:
                # make a sorting view View
                v_drift = vv.TabLayoutItem(
                    label=f"Drift map", view=vv.Image(image_path=str(fig_drift_folder / f"{recording_name}_drift.png"))
                )

            # plot motion
            v_motion = None
            if motion_folder.is_dir():
                print("\tVisualizing motion")
                motion_info = spre.load_motion_info(motion_folder)

                cmap = visualization_params["motion"]["cmap"]
                scatter_decimate = visualization_params["motion"]["scatter_decimate"]
                figsize = visualization_params["motion"]["figsize"]

                fig_motion = plt.figure(figsize=figsize)
                w_motion = sw.plot_motion(
                    motion_info,
                    recording=recording,
                    figure=fig_motion,
                    color_amplitude=True,
                    amplitude_cmap=cmap,
                    scatter_decimate=scatter_decimate,
                )
                fig_motion.savefig(fig_drift_folder / f"{recording_name}_motion.png", dpi=300)

                if not skip_timeseries:
                    # make a sorting view View
                    v_motion = vv.TabLayoutItem(
                        label=f"Motion",
                        view=vv.Image(image_path=str(fig_drift_folder / f"{recording_name}_motion.png")),
                    )

        # timeseries
        if not visualization_params["timeseries"]["skip"]:
            timeseries_tab_items = []
            print(f"\tVisualizing timeseries")

            timeseries_data = preprocessing_vizualization_data[recording_name]["timeseries"]
            recording_full_dict = timeseries_data["full"]
            recording_proc_dict = timeseries_data["proc"]

            # get random chunks to estimate clims
            clims_full = {}
            recording_full_loaded = {}
            for layer, rec_dict in recording_full_dict.items():
                try:
                    rec = si.load_extractor(rec_dict, base_folder=data_folder)
                except:
                    print(f"\t\tCould not load layer {layer}. Error:\n{e}\nSkipping")
                    continue
                chunk = si.get_random_data_chunks(rec)
                max_value = np.quantile(chunk, 0.99) * 1.2
                clims_full[layer] = (-max_value, max_value)
                recording_full_loaded[layer] = rec
            clims_proc = {}
            if recording_proc_dict is not None:
                recording_proc_loaded = {}
                for layer, rec_dict in recording_proc_dict.items():
                    try:
                        rec = si.load_extractor(rec_dict, base_folder=data_folder)
                    except:
                        print(f"\t\tCould not load layer {layer}. Skipping")
                        continue
                    chunk = si.get_random_data_chunks(rec)
                    max_value = np.quantile(chunk, 0.99) * 1.2
                    clims_proc[layer] = (-max_value, max_value)
                    recording_proc_loaded[layer] = rec
            else:
                print(f"\tPreprocessed timeseries not avaliable")

            fs = recording.sampling_frequency
            n_snippets_per_seg = visualization_params["timeseries"]["n_snippets_per_segment"]
            try:
                for segment_index in range(recording.get_num_segments()):
                    segment_duration = recording.get_num_samples(segment_index) / fs
                    # evenly distribute t_starts across segments
                    t_starts = np.linspace(0, segment_duration, n_snippets_per_seg + 2)[1:-1]
                    for t_start in t_starts:
                        time_range = np.round(
                            np.array([t_start, t_start + visualization_params["timeseries"]["snippet_duration_s"]]), 1
                        )
                        if not skip_timeseries:
                            w_full = sw.plot_timeseries(
                                recording_full_loaded,
                                order_channel_by_depth=True,
                                time_range=time_range,
                                segment_index=segment_index,
                                clim=clims_full,
                                backend="sortingview",
                                generate_url=False,
                            )
                            if recording_proc_dict is not None:
                                w_proc = sw.plot_timeseries(
                                    recording_proc_loaded,
                                    order_channel_by_depth=True,
                                    time_range=time_range,
                                    segment_index=segment_index,
                                    clim=clims_proc,
                                    backend="sortingview",
                                    generate_url=False,
                                )
                                view = vv.Splitter(
                                    direction="horizontal",
                                    item1=vv.LayoutItem(w_full.view),
                                    item2=vv.LayoutItem(w_proc.view),
                                )
                            else:
                                view = w_full.view
                            v_item = vv.TabLayoutItem(
                                label=f"Timeseries - Segment {segment_index} - Time: {time_range}", view=view
                            )
                            timeseries_tab_items.append(v_item)
                if not skip_timeseries:
                    # add drift map
                    timeseries_tab_items.append(v_drift)

                    # add motion if available
                    if v_motion is not None:
                        timeseries_tab_items.append(v_motion)

                    v_timeseries = vv.TabLayout(items=timeseries_tab_items)
                    try:
                        url = v_timeseries.url(label=f"{session_name} - {recording_name}")
                        print(f"\n{url}\n")
                        visualization_output["timeseries"] = url
                    except Exception as e:
                        print("KCL error", e)
                else:
                    print(f"\tSkipping timeseries for testing")
            except Exception as e:
                print(f"Something wrong when visualizing timeseries: {e}")

        # sorting summary
        skip_sorting_summary = True
        if analyzer_folder.is_dir():
            try:
                analyzer = si.load_sorting_analyzer(analyzer_folder)
                # analyzer.set_recording(si.load_extractor(recording_folder))
                print(f"\tVisualizing sorting summary")
                skip_sorting_summary = False
            except:
                pass

        if not skip_sorting_summary:
            unit_table_properties = []
            # add firing rate and amplitude columns
            if analyzer.has_extension("quality_metrics"):
                qm = analyzer.get_extension("quality_metrics").get_data()
                if "firing_rate" in qm.columns:
                    firing_rates = np.round(qm["firing_rate"].values, 2)
                    analyzer.sorting.set_property("firing_rate", firing_rates)
                    unit_table_properties.append("firing_rate")
                if "amplitude_median" in qm.columns:
                    unit_amplitudes = np.round(qm["amplitude_median"].values, 2)
                    analyzer.sorting.set_property("amplitude", unit_amplitudes)
                    unit_table_properties.append("amplitude")

            # add curation column
            if qc_file.is_file():
                # add qc property to analyzer sorting
                default_qc = np.load(qc_file)
                analyzer.sorting.set_property("default_qc", default_qc)
                unit_table_properties.append("default_qc")

            # add noise decoder column
            if unit_classifier_file.is_file():
                # add decoder_label and decoder probability
                unit_classifier_df = pd.read_csv(unit_classifier_file, index_col=False)
                decoder_label = unit_classifier_df["decoder_label"]
                analyzer.sorting.set_property("decoder_label", decoder_label)
                unit_table_properties.append("decoder_label")
                decoder_prob = np.round(unit_classifier_df["decoder_probability"], 2)
                analyzer.sorting.set_property("decoder_prob", decoder_prob)
                unit_table_properties.append("decoder_prob")

            # retrieve sorter name (if spike sorting was performed)
            data_process_spikesorting_json = spikesorted_folder / f"data_process_spikesorting_{recording_name}.json"
            if data_process_spikesorting_json.is_file():
                with open(data_process_spikesorting_json, "r") as f:
                    data_process_spikesorting = json.load(f)
                    sorter_name = data_process_spikesorting["parameters"]["sorter_name"]
            else:
                sorter_name = "unknown"

            if len(analyzer.unit_ids) > 0:
                # tab layout with Summary and Quality Metrics
                v_qm = sw.plot_quality_metrics(
                    analyzer,
                    skip_metrics=["isi_violations_count", "rp_violations"],
                    include_metrics_data=True,
                    backend="sortingview",
                    generate_url=False,
                ).view
                v_sorting = sw.plot_sorting_summary(
                    analyzer, unit_table_properties=unit_table_properties, curation=True, 
                    label_choices=LABEL_CHOICES, backend="sortingview", generate_url=False
                ).view

                v_summary = vv.TabLayout(
                    items=[
                        vv.TabLayoutItem(label="Sorting summary", view=v_sorting),
                        vv.TabLayoutItem(label="Quality Metrics", view=v_qm),
                    ]
                )

                try:
                    # pre-generate gh for curation
                    gh_path = f"{GH_CURATION_REPO}/{session_name}/{recording_name}/{sorter_name}/curation.json"
                    state = dict(sortingCuration=gh_path)
                    url = v_summary.url(
                        label=f"{session_name} - {recording_name} - {sorter_name} - Sorting Summary", state=state
                    )
                    print(f"\n{url}\n")
                    visualization_output["sorting_summary"] = url

                except Exception as e:
                    print("KCL error", e)
            else:
                print("\tSkipping sorting summary visualization for {recording_name}. No units after curation.")
        else:
            print(f"\tSkipping sorting summary visualization for {recording_name}. No sorting information available.")

        # save params in output
        visualization_notes = json.dumps(visualization_output, indent=4)
        # replace special characters
        visualization_notes = visualization_notes.replace('\\"', "%22")
        visualization_notes = visualization_notes.replace("#", "%23")

        # remove escape characters
        visualization_output_file.write_text(visualization_notes)

        # save vizualization output
        t_visualization_end = time.perf_counter()
        elapsed_time_visualization = np.round(t_visualization_end - t_visualization_start, 2)

        visualization_params["recording_name"] = recording_name
        visualization_process = DataProcess(
            name="Ephys visualization",
            software_version=VERSION,  # either release or git commit
            start_date_time=datetime_start_visualization,
            end_date_time=datetime_start_visualization + timedelta(seconds=np.floor(elapsed_time_visualization)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=visualization_params,
            outputs=visualization_output,
            notes=visualization_notes,
        )
        with open(visualization_output_process_json, "w") as f:
            f.write(visualization_process.model_dump_json(indent=3))

    # save vizualization output
    t_visualization_end_all = time.perf_counter()
    elapsed_time_visualization_all = np.round(t_visualization_end_all - t_visualization_start_all, 2)

    print(f"VISUALIZATION time: {elapsed_time_visualization_all}s")
