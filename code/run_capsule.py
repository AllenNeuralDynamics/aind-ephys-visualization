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
    else:
        postprocessed_folder = data_folder
        preprocessed_folder = data_folder
        curation_folder = data_folder
        unit_classifier_folder = data_folder
        spikesorted_folder = data_folder
        data_processes_spikesorting_folder = data_folder

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

    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files")

    # loop through block-streams
    for recording_name in recording_names:
        t_visualization_start = time.perf_counter()
        datetime_start_visualization = datetime.now()
        visualization_output = {}

        recording_folder = preprocessed_folder / f"preprocessed_{recording_name}"
        analyzer_binary_folder = postprocessed_folder / f"postprocessed_{recording_name}"
        analyzer_zarr_folder = postprocessed_folder / f"postprocessed_{recording_name}.zarr"
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

        recording_job_dict = None
        for job_dict in job_dicts:
            if recording_name in job_dict["recording_name"]:
                print("\tFound JSON file associated to recording")
                recording_job_dict = job_dict
                break

        if recording_job_dict is not None:
            skip_times = recording_job_dict.get("skip_times", False)
        else:
            skip_times = False

        # use spike locations
        skip_drift = False
        spike_locations_available = False
        # use spike locations
        analyzer_folder = None
        if analyzer_binary_folder.is_dir():
            analyzer_folder = analyzer_binary_folder
        elif analyzer_zarr_folder.is_dir():
            analyzer_folder = analyzer_zarr_folder

        if analyzer_folder is not None:
            try:
                analyzer = si.load_sorting_analyzer(analyzer_folder)
                # here recording_folder MUST exist
                assert recording_folder.is_dir(), f"Recording folder {recording_folder} does not exist"
                recording = si.load_extractor(recording_folder)
                if skip_times:
                    recording.reset_times()
                if analyzer.has_extension("spike_locations"):
                    print(f"\tVisualizing drift maps using spike sorted data")
                    spike_locations_available = True
            except Exception as e:
                print(
                    f"\tCould not load sorting analyzer or recording for {recording_name}: Error:\n{e}"
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
                if skip_times:
                    recording.reset_times()

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
                print(f"\t\tDetected {len(peaks)} peaks")
                peak_amps = peaks["amplitude"]
                if len(peaks) == 0:
                    print("\t\tNo peaks detected. Skipping drift map")
                    skip_drift = True
            except Exception as e:
                print(f"\t\tCould not load drift recording. Error:\n{e}\nSkipping")
                skip_drift = True

        if not skip_drift:
            fig_drift, axs_drift = plt.subplots(
                ncols=recording.get_num_segments(), figsize=visualization_params["drift"]["figsize"]
            )
            y_locs = recording.get_channel_locations()[:, 1]
            depth_lim = [np.min(y_locs), np.max(y_locs)]

            for segment_index in range(recording.get_num_segments()):
                if recording.get_num_segments() == 1:
                    ax_drift = axs_drift
                else:
                    ax_drift = axs_drift[segment_index]
                if spike_locations_available:
                    sorting_analyzer_to_plot = analyzer
                    peaks_to_plot = None
                    peak_locations_to_plot = None
                    sampling_frequency = None
                else:
                    sorting_analyzer_to_plot = None
                    peaks_to_plot = peaks
                    peak_locations_to_plot = peak_locations
                    sampling_frequency = recording.sampling_frequency

                _ = sw.plot_drift_raster_map(
                    sorting_analyzer=sorting_analyzer_to_plot,
                    peaks=peaks_to_plot,
                    peak_locations=peak_locations_to_plot,
                    sampling_frequency=sampling_frequency,
                    segment_index=segment_index,
                    depth_lim=depth_lim,
                    clim=(visualization_params["drift"]["vmin"], visualization_params["drift"]["vmax"]),
                    cmap=visualization_params["drift"]["cmap"],
                    scatter_decimate=visualization_params["drift"]["n_skip"],
                    alpha=visualization_params["drift"]["alpha"],
                    ax=ax_drift
                )
                ax_drift.spines["top"].set_visible(False)
                ax_drift.spines["right"].set_visible(False)

            fig_drift_folder = scratch_folder / "drift_maps"
            fig_drift_folder.mkdir(exist_ok=True)
            fig_drift.savefig(fig_drift_folder / f"{recording_name}_drift.png", dpi=300)

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
                # motion correction is performed after concatenation
                # since multi-segment is not supported
                if recording.get_num_segments() > 1:
                    recording_c = si.concatenate_recordings([recording])
                else:
                    recording_c = recording
                w_motion = sw.plot_motion_info(
                    motion_info,
                    recording=recording_c,
                    figure=fig_motion,
                    color_amplitude=True,
                    amplitude_cmap=cmap,
                    scatter_decimate=scatter_decimate,
                )
                fig_motion.savefig(fig_drift_folder / f"{recording_name}_motion.png", dpi=300)

                # make a sorting view View
                v_motion = vv.TabLayoutItem(
                    label=f"Motion",
                    view=vv.Image(image_path=str(fig_drift_folder / f"{recording_name}_motion.png")),
                )

        # timeseries
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
                if skip_times:
                    rec.reset_times()
            except Exception as e:
                print(f"\t\tCould not load layer {layer}. Error:\n{e}\nSkipping")
                continue
            chunk = si.get_random_data_chunks(rec)
            max_value = np.quantile(chunk, 0.99) * 1.2
            clims_full[layer] = (-max_value, max_value)
            # explicitly set timestamps if not present
            for segment_index in range(rec.get_num_segments()):
                if not rec.has_time_vector(segment_index=segment_index):
                    times = rec.get_times(segment_index=segment_index)
                    rec.set_times(times, segment_index=segment_index, with_warning=False)
            recording_full_loaded[layer] = rec
        clims_proc = {}
        if recording_proc_dict is not None:
            recording_proc_loaded = {}
            for layer, rec_dict in recording_proc_dict.items():
                try:
                    rec = si.load_extractor(rec_dict, base_folder=data_folder)
                    if skip_times:
                        rec.reset_times()
                except:
                    print(f"\t\tCould not load layer {layer}. Skipping")
                    continue
                chunk = si.get_random_data_chunks(rec)
                max_value = np.quantile(chunk, 0.99) * 1.2
                clims_proc[layer] = (-max_value, max_value)
                # explicitly set timestamps if not present
                for segment_index in range(rec.get_num_segments()):
                    if not rec.has_time_vector(segment_index=segment_index):
                        times = rec.get_times(segment_index=segment_index)
                        rec.set_times(times, segment_index=segment_index, with_warning=False)
                recording_proc_loaded[layer] = rec
        else:
            print(f"\tPreprocessed timeseries not avaliable")

        fs = recording.sampling_frequency
        n_snippets_per_seg = visualization_params["timeseries"]["n_snippets_per_segment"]
        # try:
        for segment_index in range(recording.get_num_segments()):
            # evenly distribute t_starts across segments
            times = recording.get_times(segment_index=segment_index)
            t_starts = np.linspace(times[0], times[-1], n_snippets_per_seg + 2)[1:-1]
            for t_start in t_starts:
                time_range = np.round(
                    np.array([t_start, t_start + visualization_params["timeseries"]["snippet_duration_s"]]), 1
                )
                w_full = sw.plot_traces(
                    recording_full_loaded,
                    order_channel_by_depth=True,
                    time_range=time_range,
                    segment_index=segment_index,
                    clim=clims_full,
                    backend="sortingview",
                    generate_url=False,
                )
                if recording_proc_dict is not None:
                    w_proc = sw.plot_traces(
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
            print(f"Figurl-Sortingview plotting error: {e}")

        # sorting summary
        skip_sorting_summary = True
        if analyzer_folder is not None:
            try:
                analyzer = si.load_sorting_analyzer(analyzer_folder)
                print(f"\tVisualizing sorting summary")
                skip_sorting_summary = False
            except:
                pass

        if not skip_sorting_summary:
            unit_table_properties = []
            # add firing rate and amplitude columns
            if analyzer.has_extension("quality_metrics"):
                qm = analyzer.get_extension("quality_metrics").get_data()
                unit_table_properties.append("firing_rate")

            amplitudes = si.get_template_extremum_amplitude(analyzer, mode="peak_to_peak")
            analyzer.sorting.set_property("amplitude", list(amplitudes.values()))
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
