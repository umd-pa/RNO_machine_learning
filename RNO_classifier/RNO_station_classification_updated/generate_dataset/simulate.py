import argparse
import numpy as np
import os
import functools
import datetime as dt

from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing, fft

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector.response import Response

import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold
from NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger import BeamformedPowerIntegrationTrigger

import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.phasedArrayTrigger

import logging
import matplotlib.pyplot as plt
logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
logger.setLevel(logging.INFO)


deep_trigger_channels = np.array([0, 1, 2, 3])
noised_channels = np.array([30, 31, 32, 33])
noiseless_channels = np.array([40, 41 ,42 ,43])
trace_channels = np.concatenate([deep_trigger_channels, noised_channels, noiseless_channels])

def get_abs_path(rel_path):
    """
    Converts a relative path to an absolute path based on
    the location of THIS script.
    """
    if rel_path is None:
        return None
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(caching=False, pre_pulse_time=400 * units.ns)

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogHardwareResponse.begin(trigger_channels=deep_trigger_channels)

highLowThreshold = highLowThreshold.triggerSimulator()
rnogADCResponse = triggerBoardResponse.triggerBoardResponse(log_level=logging.INFO)
rnogADCResponse.begin(clock_offset=0, adc_output="counts")
phasedArrayTrigger = NuRadioReco.modules.phasedarray.phasedArrayTrigger.PhasedArrayTrigger()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

beamformedPowerIntegrationTrigger = BeamformedPowerIntegrationTrigger()
beamformedPowerIntegrationTrigger.begin()

main_low_angle = np.deg2rad(-60)
main_high_angle = np.deg2rad(60)
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 12))
power_trigger_kwargs = {
    "phasing_angles": phasing_angles,
    "ref_index": 1.75,
    "apply_digitization": False,
    "adc_output": "counts",
    "upsampling_factor": 4,
    "upsampling_method": "fir",
    "window": 24,
    "averaging_divisor": 32,
    "step": 8,
    "saturation_bits": 8,
    "filter_taps":45,
    "coeff_gain": 256
}

envelope_trigger_kwargs = {
    "phasing_angles": phasing_angles,
    "ref_index": 1.75,
    "trigger_adc": True,
    "trigger_filter": None,
    "apply_digitization": False,
    "adc_output": "counts",
    "upsampling_factor": 1,
    "upsampling_method": "fir",
    "saturation_bits": 8,
    "filter_taps":45,
    "coeff_gain": 256,
    "ideal_transformer": False
}


def detector_simulation(evt, station, det, noise_vrms, max_freq, add_noise=True):
    """ Run the detector simulation.

    It performs the following steps:
    - efieldToVoltageConverter: Convert the electric fields to voltages
    - channelGenericNoiseAdder: Add noise to the channels
    - rnogHardwareResponse: Apply the hardware response (for RADIANT and FLOWER channels)

    Parameters
    ----------
    evt : NuRadioMC.framework.event.Event
        The event to simulate the detector response for.
    station : NuRadioMC.framework.station.Station
        The station to simulate the detector response for.
    det : NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector description.
    noise_vrms : float or dict of list
        The noise vrms (without any filter!). If a dict is given, the keys are the channel ids.
    max_freq : float
        The maximum frequency for the noise, i.e., the nyquist frequency for the simulated sampling rate.
    """
    efieldToVoltageConverter.run(evt, station, det, channel_ids=trace_channels)

    if add_noise:
        channelGenericNoiseAdder.run(
            evt, station, det, amplitude=noise_vrms, min_freq=0 * units.MHz,
            max_freq=max_freq, type='rayleigh', 
            excluded_channels=noiseless_channels
            )

    rnogHardwareResponse.run(evt, station, det, sim_to_data=True)


def rnog_flower_board_high_low_trigger_simulations(
        evt, station, det, trigger_channels, trigger_channel_noise_vrms, high_low_trigger_thresholds=None,
        pa_power_trigger_threshold=None):
    """ Run the RNO-G FLOWER board high-low trigger simulations.

    This function runs the RNO-G FLOWER board high-low trigger simulations. It performs the following steps:
    - rnogADCResponse: Digitize the traces and run the FLOWER board response
    - highLowThreshold: Apply the high-low trigger thresholds

    Parameters
    ----------
    evt : NuRadioMC.framework.event.Event
        The event to simulate the detector response for.
    station : NuRadioMC.framework.station.Station
        The station to simulate the detector response for.
    det : NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector description.
    trigger_channels : list
        The trigger channels (FLOWER) to simulate.
    trigger_channel_noise_vrms : list
        The noise vrms for the trigger channels.
    high_low_trigger_thresholds : dict
        The high-low trigger thresholds for the different trigger rates.

    Returns
    -------
    list
        The noise vrms after the gain for the trigger channels.
    """
    # Runs the FLOWER board response
    vrms_after_gain = rnogADCResponse.run(
        evt, station, det, trigger_channels=trigger_channels,
        vrms=trigger_channel_noise_vrms, digitize_trace=True,
    )

    for idx, trigger_channel in enumerate(trigger_channels):
        if rnogADCResponse.adc_output == "counts":
            logger.debug(
                'Vrms = {:.2f} mV (before flower gain) / {:.2f} ADC (after flower gain).'.format(
                    trigger_channel_noise_vrms[idx] / units.mV, vrms_after_gain[idx]
            ))
        else:
            logger.debug(
                'Vrms = {:.2f} mV (before flower gain) / {:.2f} mV (after flower gain).'.format(
                    trigger_channel_noise_vrms[idx] / units.mV, vrms_after_gain[idx] / units.mV
                ))

    # this is only returning the correct value if digitize_trace=True for self.rnogADCResponse.run(..)
    flower_sampling_rate = station.get_trigger_channel(trigger_channels[0]).get_sampling_rate()
    logger.debug('Flower sampling rate is {:.1f} MHz'.format(
        flower_sampling_rate / units.MHz
    ))

    if high_low_trigger_thresholds is not None:
        for thresh_key, threshold in high_low_trigger_thresholds.items():

            if rnogADCResponse.adc_output == "voltage":
                threshold_high = {channel_id: threshold * vrms for channel_id, vrms
                    in zip(trigger_channels, vrms_after_gain)}
                threshold_low = {channel_id: -1 * threshold * vrms for channel_id, vrms
                    in zip(trigger_channels, vrms_after_gain)}
            else:
                # We round here. This is not how an ADC works but I think this is not needed here.
                threshold_high = {channel_id: int(round(threshold * vrms)) for channel_id, vrms
                    in zip(trigger_channels, vrms_after_gain)}
                threshold_low = {channel_id: int(round(-1 * threshold * vrms)) for channel_id, vrms
                    in zip(trigger_channels, vrms_after_gain)}

            highLowThreshold.run(
                evt, station, det,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                use_digitization=False, #the trace has already been digitized with the rnogADCResponse
                high_low_window=6 / flower_sampling_rate,
                coinc_window=20 / flower_sampling_rate,
                number_concidences=2,
                triggered_channels=trigger_channels,
                trigger_name=f"deep_high_low_{thresh_key}",
                pre_trigger_time=250 * units.ns,
            )

            station.get_trigger(f"deep_high_low_{thresh_key}").set_jitter_params(
                sample_block_size=128,
                gaussian_jitter=0
            )

    if pa_power_trigger_threshold is not None:
        for thresh_key, threshold_vrms in pa_power_trigger_threshold.items():

            if rnogADCResponse.adc_output == "voltage":
                threshold = {channel_id: threshold_vrms * vrms for channel_id, vrms
                    in zip(trigger_channels, vrms_after_gain)}
            else:
                # We round here. This is not how an ADC works but I think this is not needed here.
                threshold = {channel_id: int(round(threshold_vrms * vrms)) for channel_id, vrms
                    in zip(trigger_channels, vrms_after_gain)}


            print("beamformed trigger channels:", deep_trigger_channels)

            for ch in deep_trigger_channels:
                ch = int(ch)

                try:
                    sr_reg = station.get_channel(ch).get_sampling_rate() / units.MHz
                except Exception as e:
                    sr_reg = f"missing ({e})"

                try:
                    sr_trig = station.get_trigger_channel(ch).get_sampling_rate() / units.MHz
                except Exception as e:
                    sr_trig = f"missing ({e})"

                print(f"ch {ch}: regular = {sr_reg} MHz, trigger = {sr_trig} MHz")
            pa_power_threshold = np.round(np.sum(np.array(list(threshold.values()))**2))
            beamformedPowerIntegrationTrigger.run(
                evt, station, det,
                triggered_channels=deep_trigger_channels,
                trigger_name=f"pa_power_{thresh_key}",
                threshold=pa_power_threshold,
                **power_trigger_kwargs
            )

            station.get_trigger(f"pa_power_{thresh_key}").set_jitter_params(
                sample_block_size=128,
                gaussian_jitter=0
            )


    return vrms_after_gain


@functools.lru_cache(maxsize=128)  # this is dangerous if the detector changes it will not notice it!
def get_response_conversion(det, station_id, channel_id, gain_in_dB=3.5):
    """ Get the response conversion between DAQ (RADIANT) and trigger (FLOWER) channel for the given station and channel. """
    radiant_channel = det.get_signal_chain_response(station_id, channel_id, trigger=False)
    flower_channel = det.get_signal_chain_response(station_id, channel_id, trigger=True)

    # radiant = radiant_channel.get("radiant_response")
    radiant_coax = radiant_channel.get("coax_cable")

    flower = flower_channel.get("radiant_response")  # yep we use the same collection name...
    flower_coax = flower_channel.get("coax_cable")

    # we are not using the radiant because we would devide by ~0 out of band
    conversion = flower * flower_coax / radiant_coax

    if gain_in_dB is not None:
        freqs = np.arange(10, 1200, 1) * units.MHz
        gain = np.full_like(freqs, gain_in_dB)
        phase = np.zeros_like(freqs)
        fake_radiant = Response(freqs, [gain, phase], ["dB", "rad"], name="fake_radiant", station_id=-1, channel_id=-1)
        conversion *= fake_radiant

    return conversion


def get_vrms_from_temperature_for_trigger_channels(det, station_id, trigger_channels, temperature):
    """ Get the vrms from the temperature for the trigger channels. """
    vrms_per_channel = []
    for channel_id in trigger_channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)
        vrms_per_channel.append(
            signal_processing.calculate_vrms_from_temperature(temperature=temperature, response=resp)
        )

    return np.array(vrms_per_channel)


class mySimulation(simulation.simulation):

    def __init__(self, threshold=1.0, trig_chan=0, station_id=None, **kwargs):

        assert station_id is not None, "Please provide station_id"

        # Read config to get noise type
        tmp_config = simulation.get_config(kwargs["config_file"])

        def wrapper_detector_simulation(*args, **kwargs_inner):
            noise_vrms = signal_processing.calculate_vrms_from_temperature(
                temperature=tmp_config['trigger']['noise_temperature'],
                bandwidth=tmp_config["sampling_rate"] / 2,
            )
            logger.info(
                "NuRadioMC's Noise VRMS (before sim): %.6f mV",
                noise_vrms / units.mV,
            )
            detector_simulation(
                *args,
                **kwargs_inner,
                noise_vrms=noise_vrms,
                max_freq=tmp_config["sampling_rate"] / 2,
            )

        self._detector_simulation_part2 = wrapper_detector_simulation

        self.threshold = threshold
        self.trig_chan = trig_chan

        det = kwargs["det"]
        if getattr(det, "_det_imported_from_file", False):
            t0, t1 = det._time_periods_per_station[station_id]["modification_timestamps"]
            kwargs["evt_time"] = t0 + (t1 - t0) / 2

        print(f"Using evt_time = {kwargs['evt_time']}")
        det.update(kwargs["evt_time"])

        super().__init__(**kwargs)  # this calls self._det.update(self._evt_time) internally

        self.high_low_trigger_thresholds = {"sigma3.7": 3.7,}
        self.pa_power_trigger_threshold = {"sigma3.0": 3.0}

        # computed here (after super().__init__) so it uses the detector as updated by the base class,
        # rather than needing a separate det.update(event_time) call up in __main__
        self.trigger_channel_noise_vrms = get_vrms_from_temperature_for_trigger_channels(
            self._det, station_id, deep_trigger_channels, self._config['trigger']['noise_temperature'])

        print(f"Running with threshold: {self.threshold}, trig chan {self.trig_chan}")

    def _detector_simulation_filter_amp(self, evt, station, det):
        # apply the amplifiers and filters to get to RADIANT-level
        rnogHardwareResponse.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        vrms_after_gain = rnog_flower_board_high_low_trigger_simulations(
            evt, station, det, deep_trigger_channels, self.trigger_channel_noise_vrms,
            self.high_low_trigger_thresholds, self.pa_power_trigger_threshold
        )

        for idx, trigger_channel in enumerate(deep_trigger_channels):
            self._Vrms_per_trigger_channel[station.get_id()][trigger_channel] = vrms_after_gain[idx]

        threshold_value = self.threshold * self._Vrms_per_channel[station.get_id()][self.trig_chan]
        print(f"Channel 0 trigger channel VRMS = {vrms_after_gain[0]} mV")
        print(f"Channel 0 regular RADIANT channel VRMS = {self._Vrms_per_channel[station.get_id()][0]} mV")

        simpleThreshold.run(evt, station, det,
                             threshold=threshold_value,
                             triggered_channels=[self.trig_chan],
                             number_concidences=1,
                             pre_trigger_time=200 * units.ns,
                             trigger_name='tuned_threshold')
        station.get_trigger('tuned_threshold').set_primary(True)

        internal_noise_vrms = self._Vrms_per_channel[station.get_id()][self.trig_chan]
        internal_noise_vrms_30 = self._Vrms_per_channel[station.get_id()][int(noised_channels[0])]

        # RNO-G phased array. Trigger threshold calculated using T01MeasureNoiselevel.py in phased array
        phasedArrayTrigger.run(evt, station, det,
                                threshold=38 * np.power(internal_noise_vrms, 2.0),
                                triggered_channels=noiseless_channels,
                                trigger_name='rnog_phased_array_1.7Hz',
                                apply_digitization=False  # We already handle digitization later
                                )

        if not station.has_triggered('tuned_threshold'):
            return

        ch40_trace = station.get_channel(40).get_trace().astype(np.float32)
        ch30_trace = station.get_channel(30).get_trace().astype(np.float32)

        def est_noise_vrms(trace):
            half = len(trace) // 2
            peak_idx = np.argmax(np.abs(trace))
            signal_indices = np.arange(peak_idx - half // 2,
                                        peak_idx - half // 2 + half) % len(trace)
            mask = np.ones(len(trace), dtype=bool)
            mask[signal_indices] = False
            noise_samples = trace[mask]
            return np.sqrt(np.mean(noise_samples ** 2))
        

        ch40_max_amp = np.max(ch40_trace)
        ch30_max_amp = np.max(ch30_trace)

        logger.info(
            "Ch40 NuRadioMC's Noise VRMS (during sim): %.6f mV",
            internal_noise_vrms / units.mV,
        )
        logger.info(
            "Ch40 Amplitude Threshold of tuned_threshold from trig chan VRMS: %.6f mV, Max amplitude: %.6f mV",
            threshold_value / units.mV,
            ch40_max_amp / units.mV,
        )
        logger.info(
            "Ch30 (NOISED) max amplitude: %.6f mV",
            ch30_max_amp / units.mV,
        )
        logger.info(
            "Ch30 NuRadioMC's Noise VRMS: %.6f mV, Actual Noise VRMS: %.6f mV",
            internal_noise_vrms_30 / units.mV,
            est_noise_vrms(ch30_trace) / units.mV,
        )

        for trig_name, trig in station.get_triggers().items():
            print(f"{trig_name}: triggered={trig.has_triggered()}")


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to RNO-G detector description file (selects a single station)')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
parser.add_argument('--threshold', type=float, required=True,
                    help='the threshold')
parser.add_argument('--trig_chan', type=int, required=True,
                    help='the trigger channel')
parser.add_argument('--hw_path', type=str, default=None, required=False,
                    help="unused for RNO-G (hardware response comes from the detector db); "
                         "kept so this script accepts the same flags as the ARA pipeline")
parser.add_argument('--station_id', default=11, type=int, required=False,
                    help="RNO-G station id; needed to build the detector and to look up each "
                         "trigger channel's signal-chain response for the FLOWER-board Vrms")
args = parser.parse_args()

if __name__ == "__main__":
    print(args.threshold)

    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=True, select_stations=args.station_id, over_write_handset_values={
            "is_noiseless": {
                0: False,
                1: False,
                2: False,
                3: False,
                30: False,
                31: False,
                32: False,
                33: False,
                40: True,
                41: True,
                42: True,
                43: True
            }
        })

    # no manual det.update(event_time) here -- mySimulation's super().__init__() handles it

    sim = mySimulation(inputfilename=get_abs_path(args.inputfilename),
                                outputfilename=get_abs_path(args.outputfilename),
                                det=det,
                                outputfilenameNuRadioReco=get_abs_path(args.outputfilenameNuRadioReco),
                                config_file=get_abs_path(args.config),
                                file_overwrite=True,
                                threshold=args.threshold,
                                trig_chan=args.trig_chan,
                                station_id=args.station_id,
                                trigger_channels=trace_channels,
                                use_cpp=True,
                                )
    logger.info(f"Trigger channel noise vrms (used for the definition of the trigger threshold): "
                f"{np.around(sim.trigger_channel_noise_vrms / units.mV, 2)} mV")
    sim.run()