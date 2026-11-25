import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelVariant {
  final String id;
  final String name;
  final String precisionTag;
  final String assetPath;
  final double sizeMb;

  const ModelVariant({
    required this.id,
    required this.name,
    required this.precisionTag,
    required this.assetPath,
    required this.sizeMb,
  });
}

class BenchmarkResult {
  final ModelVariant variant;
  final String topLabel;
  final double confidencePercent;
  final double meanLatencyMs;
  final double p95LatencyMs;
  final double fps;
  final double rssMemoryMb;

  BenchmarkResult({
    required this.variant,
    required this.topLabel,
    required this.confidencePercent,
    required this.meanLatencyMs,
    required this.p95LatencyMs,
    required this.fps,
    required this.rssMemoryMb,
  });
}

class EdgeProfilerService {
  static const List<ModelVariant> variants = [
    ModelVariant(
      id: 'fp32',
      name: 'MobileNetV2 FP32',
      precisionTag: 'FP32 Baseline',
      assetPath: 'assets/models/mobilenet_v2_fp32.tflite',
      sizeMb: 13.35,
    ),
    ModelVariant(
      id: 'fp16',
      name: 'MobileNetV2 FP16',
      precisionTag: 'FP16 Quantized',
      assetPath: 'assets/models/mobilenet_v2_fp16.tflite',
      sizeMb: 6.73,
    ),
    ModelVariant(
      id: 'int8',
      name: 'MobileNetV2 INT8',
      precisionTag: 'INT8 Dynamic',
      assetPath: 'assets/models/mobilenet_v2_int8.tflite',
      sizeMb: 3.64,
    ),
  ];

  List<String> _labels = [];

  Future<void> loadLabels() async {
    if (_labels.isNotEmpty) return;
    final raw = await rootBundle.loadString('assets/models/labels.txt');
    _labels = raw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();
  }

  /// Decodes image, resizes to 224x224, and normalizes RGB pixels to [-1.0, 1.0]
  Future<List<List<List<List<double>>>>> preprocessImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Failed to decode selected image.');
    }

    final resized = img.copyResize(decoded, width: 224, height: 224);

    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(224, (x) {
          final pixel = resized.getPixel(x, y);
          return [
            (pixel.r / 127.5) - 1.0,
            (pixel.g / 127.5) - 1.0,
            (pixel.b / 127.5) - 1.0,
          ];
        }),
      ),
    );

    return input;
  }

  /// Benchmarks a single TFLite variant over [iterations] runs (after warm-up)
  Future<BenchmarkResult> benchmarkVariant({
    required ModelVariant variant,
    required List<List<List<List<double>>>> inputTensor,
    int warmupRuns = 5,
    int iterations = 30,
  }) async {
    await loadLabels();

    final options = InterpreterOptions()..threads = 4;
    final interpreter = await Interpreter.fromAsset(
      variant.assetPath,
      options: options,
    );

    final outputBuffer = List.generate(1, (_) => List.filled(1000, 0.0));

    // Warm-up passes to stabilize CPU frequency & cache
    for (int i = 0; i < warmupRuns; i++) {
      interpreter.run(inputTensor, outputBuffer);
    }

    final latenciesMs = <double>[];
    final stopwatch = Stopwatch();

    for (int i = 0; i < iterations; i++) {
      stopwatch.reset();
      stopwatch.start();
      interpreter.run(inputTensor, outputBuffer);
      stopwatch.stop();
      latenciesMs.add(stopwatch.elapsedMicroseconds / 1000.0);
    }

    // Read resident set size (RAM footprint) on device
    final rssMb = ProcessInfo.currentRss / (1024 * 1024);

    // Extract Top-1 class prediction
    final probs = outputBuffer[0];
    int bestIdx = 0;
    double bestScore = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > bestScore) {
        bestScore = probs[i];
        bestIdx = i;
      }
    }

    interpreter.close();

    // Compute Mean & P95 Latency
    latenciesMs.sort();
    final meanMs = latenciesMs.reduce((a, b) => a + b) / latenciesMs.length;
    final p95Index = ((latenciesMs.length - 1) * 0.95).round();
    final p95Ms = latenciesMs[p95Index];
    final fps = meanMs > 0 ? 1000.0 / meanMs : 0.0;

    final label = bestIdx < _labels.length
        ? _labels[bestIdx]
        : 'Class #$bestIdx';

    return BenchmarkResult(
      variant: variant,
      topLabel: label,
      confidencePercent: (bestScore * 100).clamp(0.0, 100.0),
      meanLatencyMs: meanMs,
      p95LatencyMs: p95Ms,
      fps: fps,
      rssMemoryMb: rssMb,
    );
  }
}
