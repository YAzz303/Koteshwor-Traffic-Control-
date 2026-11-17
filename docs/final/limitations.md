# Limitations

These results come from a simulation of Koteshwor. They show that a learned signal controller can beat simple fixed rules on the modelled junction. They don't show how it would perform at the real junction.

## Synthetic traffic

- **Generated demand:** traffic is generated, not measured. Each test uses 600, 1000 or 1400 cars over 90 minutes. Departure times follow a smooth rush-hour curve that peaks after about 15–30 minutes and ends after about 77 minutes. Each car picks its route at random: 75% go straight, 25% turn.
- **No real counts:** none of this comes from counts at Koteshwor. Real volumes, turning shares and peak shape are unknown here and may be very different.
- **One test episode per level:** each demand level was tested with a single seed, and each agent was trained once. There is no measure of how much the results would vary with other traffic patterns or training runs.

## Cars only

- **One vehicle type:** every vehicle is the same passenger car (5 m long, same acceleration and speed).
- **Missing road users:** real Koteshwor traffic includes many motorcycles, plus buses, microbuses, tempos and trucks, and pedestrians crossing the roads. None of these are modelled. The slip lane, bus stops and roadside activity that slow traffic in reality are not represented either.

## Idealised lane discipline

- **Perfect drivers:** simulated drivers keep to their lanes, stop exactly at the stop line, never block the junction, never run a red light and never stop in the road.
- **Real traffic is less orderly:** two-wheelers filter between lanes and to the front of queues, vehicles use whatever space is free, and right-of-way is often negotiated rather than taken from the signal.
- **Effect on results:** the junction's real capacity and the queue patterns the agent would see could be quite different from the model.

## Short approach roads in the map area

- **Small map:** it covers a 394 m × 445 m box. Within it the approaches are short: N 239 m, SW 251–256 m, and S (Araniko) 162 m.
- **35 m Araniko lane:** the controlled Araniko lane is only 35 m before the slip lane splits off.
- **What the agent sees:** its state and reward only cover these lanes. Queues beyond them, and anything upstream or downstream of the map, are invisible to it.
- **Hidden delay at high demand:** queues reach the edge of the map and cars wait to enter it. That time isn't part of the waiting-time metric. Counting it changes the ranking of the two DQN agents at high demand (see [README.md](README.md#results)).
- **The Araniko approach suffers most:** because the agent sees only 35 m of it, it under-serves that road at high demand. Araniko cars average 324 s of total delay under the 1000-episode DQN, against 24 s for N Ring Road cars.
- **No corridor effects:** neighbouring junctions on the Ring Road and Araniko Highway, which control how traffic arrives in reality, are not modelled.

## Plain DQN without stabilisation techniques

- **Basic Deep Q-Network:** one fully connected network (4 × 400 units), trained with plain MSE loss and a simple replay buffer.
- **No standard stabilisers:** there is no separate target network, no Double DQN, no prioritised replay, no Huber loss or gradient clipping, and no reward normalisation. These are the usual ways to make DQN training stable.
- **Signs of the missing stabilisation:**
  - Test results move between 21 and 31 s per car from one checkpoint to the next.
  - Two training episodes (599 and 798) jammed completely.
  - A different training run could give noticeably different results.
- **Design choice:** the agent design was intentionally kept unchanged for this comparison. Only the training length was varied.

## The real junction

- **Signal plan is a modelling choice:** Koteshwor is controlled by traffic police, not by a signal. The traffic light, its three-phase plan and the 10 s / 4 s timings are assumptions. SUMO also warns that the SW green phase sends two flows into the same lane.
- **Layout will change:** a flyover is planned at Koteshwor. Once built, the layout and traffic flows at street level will change, and a controller trained on today's layout would no longer apply. It would need a new map and new training.
- **Map is today's OpenStreetMap:** lane counts and turn rules come from the OSM data at the time of import and were checked only against satellite imagery.
