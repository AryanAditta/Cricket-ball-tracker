BCB ball-tracking demo: four simulated LBW reviews

Each folder holds one delivery:
  replay.mp4     the broadcast-style replay as a video (for slides)
  replay.html    the same replay, interactive (open in Chrome or Edge; works offline)
  console.html   the TV umpire's console: both camera views synced frame by frame,
                 the replay, the LBW protocol step by step, and the review log

1  OUT: 140 km/h seamer, in line, hitting middle.
   On-field NOT OUT, fielding side reviews: overturned.
2  NOT OUT: leg-spinner hits the pad in line but turns past off stump.
   On-field OUT, batting side reviews: overturned.
3  NOT OUT: pitched outside leg (would have hit, but that doesn't matter).
   The console shows the fast path: stop at pitching.
4  UMPIRE'S CALL: the ball would only clip off stump.
   On-field OUT stands and the review is retained.

How these were made: for each delivery the system rendered two simulated camera
feeds (behind the bowler, square of the wicket), found the ball in them, rebuilt
its 3D path and applied the DRS LBW protocol. In replay.html, "Ground truth"
overlays the simulator's true path for comparison.

In console.html, change "On-field decision" or "Shot offered" to see how the
decision and the review outcome change. Keep console.html, replay.html and the
console_media folder together.
