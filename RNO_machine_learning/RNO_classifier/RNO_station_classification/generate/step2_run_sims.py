from __future__ import absolute_import, division, print_function
import argparse
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
import os

def get_abs_path(rel_path):
    """
    Converts a relative path to an absolute path based on 
    the location of THIS script (create_dagman.py).
    """
    # This gets the folder where create_dagman.py lives (e.g., .../jobs/)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

class mySimulation(simulation.simulation):

    def __init__(self, threshold=1.0, trig_chan=0, **kwargs):
        self.threshold = threshold
        self.trig_chan = trig_chan
        super().__init__(**kwargs)
        print(f"Running with threshold: {self.threshold}, trig chan {self.trig_chan}")
    
    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)
        hardware_response.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):

        # simple threshold trigger
        simpleThreshold.run(evt, station, det,
                             threshold=self.threshold * self._Vrms,
                             triggered_channels=[self.trig_chan],
                             number_concidences=1,
                             pre_trigger_time=1000 * units.ns,
                             trigger_name='tuned_threshold')

        # RNO-G phased array proxy 
        # https://arxiv.org/abs/2411.12922, Fig 27
        # We anchor the trigger to channel 40 (the noiseless version of ch 0, at -100)
        # (therefore, this will never fire for signal-less sims, e.g. noise sims).
        # We'd really like this to be an average snr trigger (to match the CNN)
        # but this will work for now.
        simpleThreshold.run(evt, station, det,
                             threshold=3.5 * self._Vrms,
                             triggered_channels=[100],
                             number_concidences=1,
                             pre_trigger_time=1000 * units.ns,
                             trigger_name='rnog_proxy_3.5sigma')


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
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
args = parser.parse_args()

if __name__ == "__main__":
    print(args.threshold)
    sim = mySimulation(inputfilename=get_abs_path(args.inputfilename),
                                outputfilename=get_abs_path(args.outputfilename),
                                detectorfile=get_abs_path(args.detectordescription),
                                outputfilenameNuRadioReco=get_abs_path(args.outputfilenameNuRadioReco),
                                config_file=get_abs_path(args.config),
                                file_overwrite=True,
                                threshold=args.threshold,
                                trig_chan=args.trig_chan
                                )
    sim.run()
