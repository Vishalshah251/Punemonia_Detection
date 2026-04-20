import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class Classifier {
  static const String _modelPath = 'assets/models/mobilenetv2_dynamic.tflite';
  static const int _inputSize = 224;

  late Interpreter _interpreter;

  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset(_modelPath);
  }

  /// Returns a map with keys: label (String), confidence (double 0–1)
  Map<String, dynamic> predict(File imageFile) {
    // 1. Decode and resize to 224×224 grayscale
    final raw = img.decodeImage(imageFile.readAsBytesSync())!;
    final resized = img.copyResize(raw, width: _inputSize, height: _inputSize);
    final grayscale = img.grayscale(resized);

    // 2. Build input tensor [1, 224, 224, 1] normalized 0–1
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) => [grayscale.getPixel(x, y).r / 255.0],
        ),
      ),
    );

    // 3. Run inference
    final output = [[0.0]];
    _interpreter.run(input, output);

    final confidence = output[0][0];
    final label = confidence >= 0.5 ? 'PNEUMONIA' : 'NORMAL';

    return {'label': label, 'confidence': confidence};
  }

  void dispose() {
    _interpreter.close();
  }
}
