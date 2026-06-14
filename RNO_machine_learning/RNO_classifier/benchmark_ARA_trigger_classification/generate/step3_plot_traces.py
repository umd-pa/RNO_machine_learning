from matplotlib import pyplot as plt

import NuRadioReco.modules.io.eventReader
event_reader = NuRadioReco.modules.io.eventReader.eventReader()

file = 'test_noise.nur'
event_reader.begin(file)
for iE, event in enumerate(event_reader.run()):
    primary = event.get_primary()

    for iStation, station in enumerate(event.get_stations()):

        # a fig and axes for our waveforms (4 rows, 2 columns)
        fig, axs = plt.subplots(4, 2, figsize=(12, 8))
        
        # this loops through "mock data" (with noise added, etc.) - channels 0-3
        for ch in station.iter_channels():
            ch_id = ch.get_id()
            if ch_id in [0, 1, 2, 3]:
                volts = ch.get_trace()
                times = ch.get_times()
                axs[ch_id, 0].plot(times, volts)
                axs[ch_id, 0].set_title(f"Channel {ch_id} (noised)")
        
        # # overlay noiseless versions of 0-3 on the left with matching truth colors
        # if station.has_sim_station():
        #     sim_station = station.get_sim_station()
        #     for sim_ch in sim_station.iter_channels():
        #         sim_ch_id = sim_ch.get_id()
        #         if sim_ch_id in [0, 1, 2, 3]:
        #             volts = sim_ch.get_trace()
        #             times = sim_ch.get_times()
        #             axs[sim_ch_id, 0].plot(times, volts, '--')

        # plot noiseless 40-43 on the right with truth colors
        for ch in station.iter_channels():
            ch_id = ch.get_id()
            if ch_id in [40, 41, 42, 43]:
                volts = ch.get_trace()
                times = ch.get_times()
                row = ch_id - 40
                axs[row, 1].plot(times, volts)
                axs[row, 1].set_title(f"Channel {ch_id} (noiseless)")

        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlabel("Time [ns]")
                ax.set_ylabel("Voltage [V]")

        fig.tight_layout()
        fig.savefig(f"traces_{iE}_noise.png") # save the traces
