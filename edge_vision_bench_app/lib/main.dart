import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'services/edge_profiler.dart';

void main() {
  runApp(const EdgeBenchApp());
}

class EdgeBenchApp extends StatelessWidget {
  const EdgeBenchApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeBench Vision Profiler',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF1B365D),
        useMaterial3: true,
      ),
      home: const BenchmarkHomeScreen(),
    );
  }
}

class BenchmarkHomeScreen extends StatefulWidget {
  const BenchmarkHomeScreen({super.key});

  @override
  State<BenchmarkHomeScreen> createState() => _BenchmarkHomeScreenState();
}

class _BenchmarkHomeScreenState extends State<BenchmarkHomeScreen> {
  final EdgeProfilerService _profiler = EdgeProfilerService();
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  bool _isRunning = false;
  String _statusMessage = 'Select an image or run synthetic benchmark';
  List<BenchmarkResult> _results = [];

  Future<void> _pickImage(ImageSource source) async {
    final picked = await _picker.pickImage(source: source, maxWidth: 800);
    if (picked == null) return;
    setState(() {
      _selectedImage = File(picked.path);
      _results = [];
      _statusMessage = 'Image loaded. Tap "Run Full Benchmark" to profile.';
    });
  }

  /// Generates a normalized 1x224x224x3 tensor if user wants instant profiling without a photo
  List<List<List<List<double>>>> _buildSyntheticTensor() {
    return List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) => [
            ((x % 255) / 127.5) - 1.0,
            ((y % 255) / 127.5) - 1.0,
            (((x + y) % 255) / 127.5) - 1.0,
          ],
        ),
      ),
    );
  }

  Future<void> _runAllBenchmarks() async {
    if (_isRunning) return;
    setState(() {
      _isRunning = true;
      _results = [];
      _statusMessage = 'Preprocessing input tensor (224x224 RGB)...';
    });

    try {
      final inputTensor = _selectedImage != null
          ? await _profiler.preprocessImage(_selectedImage!)
          : _buildSyntheticTensor();

      final collected = <BenchmarkResult>[];

      for (final variant in EdgeProfilerService.variants) {
        setState(() {
          _statusMessage =
              'Profiling ${variant.name} (5 warm-up + 30 timed runs)...';
        });

        // Short UI yield so status text updates cleanly
        await Future.delayed(const Duration(milliseconds: 60));

        final res = await _profiler.benchmarkVariant(
          variant: variant,
          inputTensor: inputTensor,
          warmupRuns: 5,
          iterations: 30,
        );
        collected.add(res);
        setState(() {
          _results = List.from(collected);
        });
      }

      setState(() {
        _statusMessage =
            'Completed 30-iteration benchmark across FP32, FP16 & INT8.';
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error: $e';
      });
    } finally {
      setState(() {
        _isRunning = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final baselineLatency = _results.isNotEmpty
        ? _results.first.meanLatencyMs
        : null;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'EdgeBench: On-Device Vision Profiler',
          style: TextStyle(fontWeight: FontWeight.w700, fontSize: 17),
        ),
        backgroundColor: const Color(0xFF1B365D),
        foregroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Input Image & Controls Card
          Card(
            elevation: 1,
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Container(
                    height: 170,
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: _selectedImage != null
                        ? ClipRRect(
                            borderRadius: BorderRadius.circular(10),
                            child: Image.file(
                              _selectedImage!,
                              fit: BoxFit.cover,
                            ),
                          )
                        : const Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.memory, size: 42, color: Colors.grey),
                              SizedBox(height: 8),
                              Text(
                                'No image selected (Synthetic tensor ready)',
                                style: TextStyle(color: Colors.black54),
                              ),
                            ],
                          ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _isRunning
                              ? null
                              : () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_outlined),
                          label: const Text('Gallery'),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _isRunning
                              ? null
                              : () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt_outlined),
                          label: const Text('Camera'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  FilledButton.icon(
                    onPressed: _isRunning ? null : _runAllBenchmarks,
                    style: FilledButton.styleFrom(
                      backgroundColor: const Color(0xFF1B365D),
                      padding: const EdgeInsets.symmetric(vertical: 13),
                    ),
                    icon: _isRunning
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.speed),
                    label: Text(
                      _isRunning
                          ? 'Benchmarking on Device...'
                          : 'Run Full Benchmark (FP32 / FP16 / INT8)',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _statusMessage,
                    style: TextStyle(
                      fontSize: 12.5,
                      color: Colors.grey.shade700,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 14),

          // Benchmark Results Section
          if (_results.isNotEmpty) ...[
            const Text(
              'Hardware Profiling Results (4 CPU Threads)',
              style: TextStyle(fontSize: 15.5, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ..._results.map(
              (res) =>
                  _ResultCard(result: res, baselineLatencyMs: baselineLatency),
            ),
          ],
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final BenchmarkResult result;
  final double? baselineLatencyMs;

  const _ResultCard({required this.result, required this.baselineLatencyMs});

  @override
  Widget build(BuildContext context) {
    final sizeReduction = ((1 - (result.variant.sizeMb / 13.35)) * 100).clamp(
      0.0,
      100.0,
    );
    final speedup = (baselineLatencyMs != null && result.meanLatencyMs > 0)
        ? baselineLatencyMs! / result.meanLatencyMs
        : 1.0;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 1.5,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  result.variant.name,
                  style: const TextStyle(
                    fontSize: 15.5,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1B365D),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 3,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1B365D).withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    '${result.variant.sizeMb.toStringAsFixed(2)} MB (-${sizeReduction.toStringAsFixed(1)}%)',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1B365D),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Top-1 Prediction: ${result.topLabel} (${result.confidencePercent.toStringAsFixed(1)}%)',
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
            ),
            const Divider(height: 18),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _MetricTile(
                  label: 'Mean Latency',
                  value: '${result.meanLatencyMs.toStringAsFixed(1)} ms',
                ),
                _MetricTile(
                  label: 'P95 Latency',
                  value: '${result.p95LatencyMs.toStringAsFixed(1)} ms',
                ),
                _MetricTile(
                  label: 'Throughput',
                  value: '${result.fps.toStringAsFixed(1)} FPS',
                ),
                _MetricTile(
                  label: 'RSS RAM',
                  value: '${result.rssMemoryMb.toStringAsFixed(0)} MB',
                ),
              ],
            ),
            if (result.variant.id != 'fp32') ...[
              const SizedBox(height: 8),
              Text(
                'Relative Speed vs FP32: ${speedup.toStringAsFixed(2)}x',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: speedup >= 1.0
                      ? Colors.green.shade700
                      : Colors.orange.shade800,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  final String label;
  final String value;

  const _MetricTile({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: const TextStyle(fontSize: 13.5, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
