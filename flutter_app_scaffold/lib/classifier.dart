import 'dart:typed_data';
import 'dart:convert';
import 'package:http/http.dart' as http;

class PredictionResult {
  final String label;
  final Uint8List heatmapBytes;

  PredictionResult({required this.label, required this.heatmapBytes});
}

class Classifier {
  static const String _baseUrl = 'http://localhost:5000';

  Future<PredictionResult> predict(Uint8List imageBytes, String filename) async {
    final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/gradcam'));
    request.files.add(http.MultipartFile.fromBytes(
      'image',
      imageBytes,
      filename: filename,
    ));

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception('API error ${response.statusCode}: ${response.body}');
    }

    final data    = jsonDecode(response.body) as Map<String, dynamic>;
    final label   = data['label'] as String;
    final heatmap = base64Decode(data['heatmap_b64'] as String);

    return PredictionResult(label: label, heatmapBytes: heatmap);
  }
}
