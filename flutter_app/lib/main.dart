import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';

void main() => runApp(const PneumoniaApp());

class PneumoniaApp extends StatelessWidget {
  const PneumoniaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pneumonia Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1565C0)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final Classifier _classifier = Classifier();
  final ImagePicker _picker = ImagePicker();

  File? _image;
  String? _label;
  double? _confidence;
  bool _loading = false;
  bool _modelReady = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _classifier.load();
    setState(() => _modelReady = true);
  }

  Future<void> _pickImage(ImageSource source) async {
    final picked = await _picker.pickImage(source: source);
    if (picked == null) return;

    setState(() {
      _image = File(picked.path);
      _label = null;
      _confidence = null;
      _loading = true;
    });

    // Run on a separate isolate to keep UI responsive
    final result = await Future(() => _classifier.predict(_image!));

    setState(() {
      _label = result['label'];
      _confidence = result['confidence'];
      _loading = false;
    });
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
        title: const Text('Pneumonia Detector'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Model status ──────────────────────────────────────────────
            if (!_modelReady)
              const Center(child: CircularProgressIndicator()),

            // ── Image preview ─────────────────────────────────────────────
            _ImageCard(image: _image, loading: _loading),
            const SizedBox(height: 24),

            // ── Pick buttons ──────────────────────────────────────────────
            Row(
              children: [
                Expanded(
                  child: _PickButton(
                    icon: Icons.photo_library,
                    label: 'Gallery',
                    enabled: _modelReady && !_loading,
                    onTap: () => _pickImage(ImageSource.gallery),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _PickButton(
                    icon: Icons.camera_alt,
                    label: 'Camera',
                    enabled: _modelReady && !_loading,
                    onTap: () => _pickImage(ImageSource.camera),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 32),

            // ── Result ────────────────────────────────────────────────────
            if (_label != null && !_loading)
              _ResultCard(label: _label!, confidence: _confidence!),
          ],
        ),
      ),
    );
  }
}

// ── Widgets ───────────────────────────────────────────────────────────────────

class _ImageCard extends StatelessWidget {
  final File? image;
  final bool loading;

  const _ImageCard({required this.image, required this.loading});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 280,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 8)],
      ),
      clipBehavior: Clip.antiAlias,
      child: loading
          ? const Center(child: CircularProgressIndicator())
          : image == null
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.x_ray, size: 72, color: Colors.grey[300]),
                    const SizedBox(height: 12),
                    Text(
                      'No image selected',
                      style: TextStyle(color: Colors.grey[400], fontSize: 16),
                    ),
                  ],
                )
              : Image.file(image!, fit: BoxFit.cover, width: double.infinity),
    );
  }
}

class _PickButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool enabled;
  final VoidCallback onTap;

  const _PickButton({
    required this.icon,
    required this.label,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: enabled ? onTap : null,
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
        disabledBackgroundColor: Colors.grey[300],
        padding: const EdgeInsets.symmetric(vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final String label;
  final double confidence;

  const _ResultCard({required this.label, required this.confidence});

  @override
  Widget build(BuildContext context) {
    final isPneumonia = label == 'PNEUMONIA';
    final color = isPneumonia ? const Color(0xFFB71C1C) : const Color(0xFF1B5E20);
    final bgColor = isPneumonia ? const Color(0xFFFFEBEE) : const Color(0xFFE8F5E9);
    final icon = isPneumonia ? Icons.warning_amber_rounded : Icons.check_circle_rounded;

    // Show the confidence score toward the predicted class
    final displayPct = isPneumonia ? confidence : (1.0 - confidence);

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3), width: 1.5),
      ),
      child: Column(
        children: [
          Icon(icon, size: 56, color: color),
          const SizedBox(height: 12),
          Text(
            label,
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: color,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Confidence: ${(displayPct * 100).toStringAsFixed(1)}%',
            style: TextStyle(fontSize: 16, color: color.withOpacity(0.8)),
          ),
          const SizedBox(height: 16),
          ClipRoundedRect(
            radius: 8,
            child: LinearProgressIndicator(
              value: displayPct,
              minHeight: 10,
              backgroundColor: color.withOpacity(0.15),
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            isPneumonia
                ? 'Possible signs of pneumonia detected.\nPlease consult a doctor.'
                : 'No signs of pneumonia detected.',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 13, color: color.withOpacity(0.75)),
          ),
        ],
      ),
    );
  }
}

// Simple helper to clip LinearProgressIndicator
class ClipRoundedRect extends StatelessWidget {
  final double radius;
  final Widget child;

  const ClipRoundedRect({required this.radius, required this.child, super.key});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(radius),
      child: child,
    );
  }
}
