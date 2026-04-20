import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';

// ── Palette ───────────────────────────────────────────────────────────────────
const _bg       = Color(0xFFF8F9FB);
const _surface  = Color(0xFFFFFFFF);
const _border   = Color(0xFFE8ECF0);
const _textPri  = Color(0xFF0F172A);
const _textSec  = Color(0xFF64748B);
const _accent   = Color(0xFF2563EB);
const _danger   = Color(0xFFDC2626);
const _success  = Color(0xFF16A34A);

void main() => runApp(const PneumoniaApp());

class PneumoniaApp extends StatelessWidget {
  const PneumoniaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PneumoScan',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: _bg,
        colorScheme: const ColorScheme.light(
          primary: _accent,
          surface: _surface,
        ),
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: _textPri, fontSize: 14),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Home Screen
// ══════════════════════════════════════════════════════════════════════════════

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  final _classifier = Classifier();
  final _picker     = ImagePicker();

  Uint8List? _imageBytes;
  String?    _label;
  Uint8List? _heatmapBytes;
  bool       _loading = false;
  String?    _error;

  late final AnimationController _fadeCtrl;
  late final Animation<double>   _fadeAnim;

  @override
  void initState() {
    super.initState();
    _fadeCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 400));
    _fadeAnim = CurvedAnimation(parent: _fadeCtrl, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    _fadeCtrl.dispose();
    super.dispose();
  }

  Future<void> _pick() async {
    final picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    _fadeCtrl.reset();
    setState(() {
      _imageBytes   = bytes;
      _label        = null;
      _heatmapBytes = null;
      _error        = null;
      _loading      = true;
    });

    try {
      final result = await _classifier.predict(bytes, picked.name);
      setState(() {
        _label        = result.label;
        _heatmapBytes = result.heatmapBytes;
        _loading      = false;
      });
      _fadeCtrl.forward();
    } catch (_) {
      setState(() {
        _error   = 'Cannot reach server. Make sure the Flask API is running on port 5000.';
        _loading = false;
      });
    }
  }

  void _reset() {
    _fadeCtrl.reset();
    setState(() {
      _imageBytes   = null;
      _label        = null;
      _heatmapBytes = null;
      _error        = null;
      _loading      = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.of(context).size.width > 860;

    return Scaffold(
      backgroundColor: _bg,
      body: Column(
        children: [
          _NavBar(
            showReset: _imageBytes != null,
            onReset: _reset,
          ),
          Expanded(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(
                    horizontal: 32, vertical: 28),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 1040),
                  child: wide
                      ? _WideLayout(
                          imageBytes:   _imageBytes,
                          label:        _label,
                          heatmapBytes: _heatmapBytes,
                          loading:      _loading,
                          error:        _error,
                          fadeAnim:     _fadeAnim,
                          onPick:       _pick,
                        )
                      : _NarrowLayout(
                          imageBytes:   _imageBytes,
                          label:        _label,
                          heatmapBytes: _heatmapBytes,
                          loading:      _loading,
                          error:        _error,
                          fadeAnim:     _fadeAnim,
                          onPick:       _pick,
                        ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Navigation Bar
// ══════════════════════════════════════════════════════════════════════════════

class _NavBar extends StatelessWidget {
  final bool      showReset;
  final VoidCallback onReset;
  const _NavBar({required this.showReset, required this.onReset});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: _surface,
      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 0),
      child: SafeArea(
        bottom: false,
        child: SizedBox(
          height: 60,
          child: Row(
            children: [
              // Logo mark
              Container(
                width: 32, height: 32,
                decoration: BoxDecoration(
                  color: _accent,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.biotech,
                    color: Colors.white, size: 18),
              ),
              const SizedBox(width: 10),
              const Text('PneumoScan',
                  style: TextStyle(
                    color: _textPri,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.3,
                  )),
              const Spacer(),
              // Status pill
              _Pill(
                color: _success,
                label: 'Model ready',
              ),
              if (showReset) ...[
                const SizedBox(width: 12),
                TextButton(
                  onPressed: onReset,
                  style: TextButton.styleFrom(
                    foregroundColor: _textSec,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                  ),
                  child: const Text('Clear',
                      style: TextStyle(fontSize: 13)),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Layouts
// ══════════════════════════════════════════════════════════════════════════════

class _WideLayout extends StatelessWidget {
  final Uint8List?   imageBytes;
  final String?      label;
  final Uint8List?   heatmapBytes;
  final bool         loading;
  final String?      error;
  final Animation<double> fadeAnim;
  final VoidCallback onPick;

  const _WideLayout({
    required this.imageBytes, required this.label,
    required this.heatmapBytes, required this.loading,
    required this.error, required this.fadeAnim,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: _UploadPanel(
            imageBytes: imageBytes,
            loading:    loading,
            onPick:     onPick,
          ),
        ),
        const SizedBox(width: 24),
        Expanded(
          child: _ResultPanel(
            label:        label,
            heatmapBytes: heatmapBytes,
            loading:      loading,
            error:        error,
            fadeAnim:     fadeAnim,
          ),
        ),
      ],
    );
  }
}

class _NarrowLayout extends StatelessWidget {
  final Uint8List?   imageBytes;
  final String?      label;
  final Uint8List?   heatmapBytes;
  final bool         loading;
  final String?      error;
  final Animation<double> fadeAnim;
  final VoidCallback onPick;

  const _NarrowLayout({
    required this.imageBytes, required this.label,
    required this.heatmapBytes, required this.loading,
    required this.error, required this.fadeAnim,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _UploadPanel(
            imageBytes: imageBytes, loading: loading, onPick: onPick),
        const SizedBox(height: 24),
        _ResultPanel(
            label: label, heatmapBytes: heatmapBytes,
            loading: loading, error: error, fadeAnim: fadeAnim),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Upload Panel
// ══════════════════════════════════════════════════════════════════════════════

class _UploadPanel extends StatelessWidget {
  final Uint8List?   imageBytes;
  final bool         loading;
  final VoidCallback onPick;

  const _UploadPanel({
    required this.imageBytes,
    required this.loading,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Label('X-RAY IMAGE'),
        const SizedBox(height: 10),

        // Image / drop zone
        GestureDetector(
          onTap: loading ? null : onPick,
          child: _Card(
            height: 320,
            child: loading
                ? const _Spinner(message: 'Analysing…')
                : imageBytes == null
                    ? _DropHint(onTap: onPick)
                    : _ImagePreview(bytes: imageBytes!),
          ),
        ),

        const SizedBox(height: 12),

        // Upload button
        SizedBox(
          height: 44,
          child: ElevatedButton(
            onPressed: loading ? null : onPick,
            style: ElevatedButton.styleFrom(
              backgroundColor: _accent,
              foregroundColor: Colors.white,
              disabledBackgroundColor: _border,
              disabledForegroundColor: _textSec,
              elevation: 0,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10)),
              textStyle: const TextStyle(
                  fontSize: 14, fontWeight: FontWeight.w600),
            ),
            child: Text(loading ? 'Analysing…' : 'Upload X-ray'),
          ),
        ),

        const SizedBox(height: 10),
        const _HintRow(),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Result Panel
// ══════════════════════════════════════════════════════════════════════════════

class _ResultPanel extends StatelessWidget {
  final String?      label;
  final Uint8List?   heatmapBytes;
  final bool         loading;
  final String?      error;
  final Animation<double> fadeAnim;

  const _ResultPanel({
    required this.label, required this.heatmapBytes,
    required this.loading, required this.error,
    required this.fadeAnim,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _Label('RESULT'),
        const SizedBox(height: 10),

        if (error != null)
          _ErrorBanner(message: error!)
        else if (loading)
          _Card(height: 320, child: const _Spinner(message: 'Running inference…'))
        else if (label == null)
          _Card(
            height: 320,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.arrow_back_rounded,
                    size: 32, color: _border),
                const SizedBox(height: 12),
                const Text('Upload an image to see results',
                    style: TextStyle(color: _textSec, fontSize: 14)),
              ],
            ),
          )
        else
          FadeTransition(
            opacity: fadeAnim,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _DiagnosisCard(label: label!),
                if (heatmapBytes != null) ...[
                  const SizedBox(height: 16),
                  _HeatmapCard(bytes: heatmapBytes!),
                ],
              ],
            ),
          ),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Cards
// ══════════════════════════════════════════════════════════════════════════════

class _DiagnosisCard extends StatelessWidget {
  final String label;
  const _DiagnosisCard({required this.label});

  @override
  Widget build(BuildContext context) {
    final isPositive = label == 'PNEUMONIA';
    final color      = isPositive ? _danger : _success;
    final bg         = isPositive
        ? const Color(0xFFFEF2F2)
        : const Color(0xFFF0FDF4);
    final icon       = isPositive
        ? Icons.warning_amber_rounded
        : Icons.check_circle_outline_rounded;
    final headline   = isPositive ? 'Pneumonia Detected' : 'No Pneumonia Detected';
    final sub        = isPositive
        ? 'Signs of pneumonia are present in this X-ray. Please consult a physician.'
        : 'No signs of pneumonia were detected. The X-ray appears normal.';

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(headline,
                    style: TextStyle(
                      color: color,
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                    )),
                const SizedBox(height: 6),
                Text(sub,
                    style: const TextStyle(
                        color: _textSec, fontSize: 13, height: 1.5)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HeatmapCard extends StatelessWidget {
  final Uint8List bytes;
  const _HeatmapCard({required this.bytes});

  @override
  Widget build(BuildContext context) {
    return _Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 14),
            child: Row(
              children: [
                const Text('GradCAM Activation Map',
                    style: TextStyle(
                      color: _textPri,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    )),
                const Spacer(),
                const Text('Regions the model focused on',
                    style: TextStyle(color: _textSec, fontSize: 12)),
              ],
            ),
          ),
          const Divider(height: 1, color: _border),

          // Heatmap
          ClipRRect(
            borderRadius: const BorderRadius.only(
              bottomLeft:  Radius.circular(12),
              bottomRight: Radius.circular(12),
            ),
            child: Image.memory(bytes, fit: BoxFit.contain),
          ),

          // Legend
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            child: Row(
              children: [
                const Text('Low',
                    style: TextStyle(color: _textSec, fontSize: 11)),
                const SizedBox(width: 8),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: Container(
                      height: 6,
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(colors: [
                          Color(0xFF0000FF),
                          Color(0xFF00FFFF),
                          Color(0xFF00FF00),
                          Color(0xFFFFFF00),
                          Color(0xFFFF0000),
                        ]),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                const Text('High',
                    style: TextStyle(color: _textSec, fontSize: 11)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Small reusable widgets
// ══════════════════════════════════════════════════════════════════════════════

/// White card with border and optional fixed height
class _Card extends StatelessWidget {
  final Widget child;
  final double? height;
  const _Card({required this.child, this.height});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(
        color: _surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _border),
      ),
      clipBehavior: Clip.antiAlias,
      child: child,
    );
  }
}

class _Label extends StatelessWidget {
  final String text;
  const _Label(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(text,
        style: const TextStyle(
          color: _textSec,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
        ));
  }
}

class _Pill extends StatelessWidget {
  final Color  color;
  final String label;
  const _Pill({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6, height: 6,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(label,
              style: TextStyle(
                  color: color, fontSize: 12, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}

class _Spinner extends StatelessWidget {
  final String message;
  const _Spinner({required this.message});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 24, height: 24,
            child: CircularProgressIndicator(
                strokeWidth: 2, color: _accent),
          ),
          const SizedBox(height: 14),
          Text(message,
              style: const TextStyle(color: _textSec, fontSize: 13)),
        ],
      ),
    );
  }
}

class _DropHint extends StatelessWidget {
  final VoidCallback onTap;
  const _DropHint({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: _accent.withOpacity(0.06),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.add_photo_alternate_outlined,
                size: 36, color: _accent),
          ),
          const SizedBox(height: 16),
          const Text('Click to upload an X-ray',
              style: TextStyle(
                  color: _textPri,
                  fontSize: 14,
                  fontWeight: FontWeight.w500)),
          const SizedBox(height: 4),
          const Text('JPG or PNG',
              style: TextStyle(color: _textSec, fontSize: 12)),
        ],
      ),
    );
  }
}

class _ImagePreview extends StatelessWidget {
  final Uint8List bytes;
  const _ImagePreview({required this.bytes});

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.memory(bytes, fit: BoxFit.contain),
        Positioned(
          bottom: 0, left: 0, right: 0,
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 8),
            color: Colors.black.withOpacity(0.35),
            child: const Center(
              child: Text('Click to change',
                  style: TextStyle(
                      color: Colors.white70, fontSize: 12)),
            ),
          ),
        ),
      ],
    );
  }
}

class _HintRow extends StatelessWidget {
  const _HintRow();

  @override
  Widget build(BuildContext context) {
    return const Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.lock_outline, size: 12, color: _textSec),
        SizedBox(width: 4),
        Text('Processed on your local server · not uploaded to the cloud',
            style: TextStyle(color: _textSec, fontSize: 11)),
      ],
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFEF3C7),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFFF59E0B).withOpacity(0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline,
              color: Color(0xFFD97706), size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(message,
                style: const TextStyle(
                    color: Color(0xFF92400E), fontSize: 13, height: 1.5)),
          ),
        ],
      ),
    );
  }
}
