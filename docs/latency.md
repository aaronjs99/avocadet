# Latency Instrumentation

Avocadet is designed with a "Freshness > Completeness" philosophy.

## Metrics
The system measures latency at these points:
1. **Acquisition**: Time packet is stamped in the camera driver.
2. **Ingestion**: Time packet arrives at `detector_node`.
3. **Inference Start**: Time worker thread picks up frame.
4. **Publish**: Time detection result is published.

## Configuration
In `config/runtime.yaml`:
```yaml
max_end_to_end_latency_ms: 100
drop_frames: true
```
If `(Now - Acquisition Time) > max_latency`, the frame is dropped immediately in the worker loop before inference starts.

## Monitoring
- **Logs**: Latency warnings are printed if thresholds are exceeded.
- **Annotated Image**: Visual lag can be estimated by comparing the annotated stream movement vs real interaction.
- **ROS 2 Tracing**: Standard ROS 2 tools can be used to trace message flow.
