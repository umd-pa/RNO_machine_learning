from NuRadioReco.modules.io.eventReader import eventReader

def get_unique_events(fPath):
    """
    Reads a .nur file and returns a list of unique, multi-station events.
    """
    event_reader = eventReader()
    event_reader.begin(fPath)
    
    events_unique = []
    iFirst = -1
    first_event = None

    for iE, event in enumerate(event_reader.run()):
        primary = event.get_primary()
        iP = primary.get_id() # type: ignore
        event.set_id(iP)

        # If we see a NEW particle ID...
        if iFirst != iP:
            # Save the PREVIOUS completed event
            if first_event is not None:
                events_unique.append(first_event)
            
            # Start tracking the NEW event
            iFirst = iP
            first_event = event
            
        else:
            # Add station to the CURRENT event
            first_event.set_station(event.get_station())  # type: ignore

    # --- THE FIX ---
    # Append the very last event after the loop finishes
    if first_event is not None:
        events_unique.append(first_event)

    return events_unique