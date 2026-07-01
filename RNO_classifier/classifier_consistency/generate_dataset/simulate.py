from __future__ import absolute_import, division, print_function
import argparse
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.ARA.hardwareResponseIncorporator
import NuRadioReco.modules.phasedarray.phasedArrayTrigger
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import numpy as np
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
hardware_response = NuRadioReco.modules.ARA.hardwareResponseIncorporator.hardwareResponseIncorporator()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.phasedArrayTrigger.PhasedArrayTrigger()

class mySimulation(simulation.simulation):

    def __init__(self, threshold=1.0, trig_chan=0, path=None, **kwargs):
        self.threshold = threshold
        self.trig_chan = trig_chan
        self.path = path
        super().__init__(**kwargs)
        print(f"Running with threshold: {self.threshold}, trig chan {self.trig_chan}")
    
    def _detector_simulation_filter_amp(self, evt, station, det):
        hardware_response.run(evt, station, det, sim_to_data=True, path=self.path)

    def _detector_simulation_trigger(self, evt, station, det):

        # simple threshold trigger, becomes the primary trigger because it always triggers first
        simpleThreshold.run(evt, station, det,
                             threshold=self.threshold * self._Vrms,
                             triggered_channels=[self.trig_chan],
                             number_concidences=1,
                             pre_trigger_time=1000 * units.ns,
                             trigger_name='tuned_threshold')

        # RNO-G phased array. Trigger threshold calculated using T01MeasureNoiselevel.py in phased array examples to have 
        # approximately 1Hz trigger rate
        phasedArrayTrigger.run(evt, station, det, 
                             threshold=38 * np.power(self._Vrms, 2.0),
                             triggered_channels=[40,41,42,43],
                             trigger_name='rnog_phased_array_1.7Hz',
                             apply_digitization=False # We already handle digitization later
                             )


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
parser.add_argument('--hw_path', type=str, default=None, required=False,
                    help="path to hardware response file with gain & phase info")
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
                                trig_chan=args.trig_chan,
                                path=args.hw_path
                                )
    sim.run()
