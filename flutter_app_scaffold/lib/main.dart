import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';

// ── Design tokens ─────────────────────────────────────────────────────────────
const _bg0      = Color(0xFF0A0E1A);   // deepest bg
const _bg1      = Color(0xFF0F1526);   // card bg
const _bg2      = Color(0xFF16203A);   // elevated card
const _accent   = Color(0xFF3B82F6);   // blue
const _accentLo = Color(0xFF1D4ED8);
const _cyan     = Color(0xFF06B6D4);
const _danger   = Color(0xFFEF4444);
const _dangerLo = Color(0xFF7F1D1D);
const _success  = Color(0xFF22C55E);
const _successLo= Color(0xFF14532D);
const _border   = Color(0xFF1E2D4A);
const _textPri  = Color(0xFFE2E8F0);
const _textSec  = Color(0xFF64748B);
const _textMut  = Color(0xFF334155);

void main() => runApp(const PneumoniaApp());

class PneumoniaApp extends StatelessWidget {
  const PneumoniaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PneumoScan AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: _bg0,
        fontFamily: 'SF Pro Display',
        colorScheme: const ColorScheme.dark(
          primary: _accent,
          surface: _bg1,
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
    with TickerProviderStateMixin {
  final _classifier = Classifier();
  final _picker     = ImagePicker();

  Uint8List? _imageBytes;
  String?    _label;
  Uint8List? _heatmapBytes;
  bool       _loading = false;
  String?    _error;

  late final AnimationController _fadeCtrl;
  late final Animation<double>   _fadeAnim;

  late final AnimationController _pulseCtrl;
  late final Animation<double>   _pulseAnim;

  late final AnimationController _scanCtrl;

  @override
  void initState() {
    super.initState();
    _fadeCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 600));
    _fadeAnim = CurvedAnimation(parent: _fadeCtrl, curve: Curves.easeOutCubic);

    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(seconds: 2))
      ..repeat(reverse: true);
    _pulseAnim = CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut);

    _scanCtrl = AnimationController(vsync: this, duration: const Duration(seconds: 2));
  }

  @override
  void dispose() {
    _fadeCtrl.dispose();
    _pulseCtrl.dispose();
    _scanCtrl.dispose();
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

    _scanCtrl.repeat();

    try {
      final result = await _classifier.predict(bytes, picked.name);
      _scanCtrl.stop();
      _scanCtrl.reset();
      setState(() {
        _label        = result.label;
        _heatmapBytes = result.heatmapBytes;
        _loading      = false;
      });
      _fadeCtrl.forward();
    } catch (_) {
      _scanCtrl.stop();
      _scanCtrl.reset();
      setState(() {
        _error   = 'Cannot reach server. Make sure the Flask API is running on port 5001.';
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
    final w    = MediaQuery.of(context).size.width;
    final wide = w > 900;

    return Scaffold(
      backgroundColor: _bg0,
      body: Stack(
        children: [
          // Background grid pattern
          Positioned.fill(child: _GridBackground()),

          Column(
            children: [
              _TopBar(showReset: _imageBytes != null, onReset: _reset),
              Expanded(
                child: SingleChildScrollView(
                  padding: EdgeInsets.symmetric(
                    horizontal: wide ? 48 : 20,
                    vertical: 28,
                  ),
                  child: Center(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 1100),
                      child: Column(
                        children: [
                          if (_imageBytes == null && !_loading) ...[
                            const SizedBox(height: 20),
                            _HeroHeader(pulseAnim: _pulseAnim),
                            const SizedBox(height: 48),
                          ],
                          if (wide)
                            _WideLayout(
                              imageBytes: _imageBytes, label: _label,
                              heatmapBytes: _heatmapBytes, loading: _loading,
                              error: _error, fadeAnim: _fadeAnim,
                              scanAnim: _scanCtrl, pulseAnim: _pulseAnim,
                              onPick: _pick,
                            )
                          else
                            _NarrowLayout(
                              imageBytes: _imageBytes, label: _label,
                              heatmapBytes: _heatmapBytes, loading: _loading,
                              error: _error, fadeAnim: _fadeAnim,
                              scanAnim: _scanCtrl, pulseAnim: _pulseAnim,
                              onPick: _pick,
                            ),
                          const SizedBox(height: 32),
                          if (_imageBytes == null) _StatsRow(),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Background grid
// ══════════════════════════════════════════════════════════════════════════════
class _GridBackground extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _GridPainter());
  }
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF1E2D4A).withOpacity(0.4)
      ..strokeWidth = 0.5;
    const step = 48.0;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
    // Radial gradient overlay
    final gradient = RadialGradient(
      center: const Alignment(-0.2, -0.5),
      radius: 1.2,
      colors: [
        const Color(0xFF1D4ED8).withOpacity(0.08),
        Colors.transparent,
      ],
    );
    final rect = Offset.zero & size;
    final grPaint = Paint()..shader = gradient.createShader(rect);
    canvas.drawRect(rect, grPaint);
  }

  @override
  bool shouldRepaint(_) => false;
}

// ══════════════════════════════════════════════════════════════════════════════
// Top Bar
// ══════════════════════════════════════════════════════════════════════════════
class _TopBar extends StatelessWidget {
  final bool showReset;
  final VoidCallback onReset;
  const _TopBar({required this.showReset, required this.onReset});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: _bg1.withOpacity(0.8),
        border: const Border(bottom: BorderSide(color: _border)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: SizedBox(
            height: 64,
            child: Row(
              children: [
                // Logo
                Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [_accent, _cyan],
                    ),
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: _accent.withOpacity(0.4),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Icon(Icons.biotech_rounded, color: Colors.white, size: 20),
                ),
                const SizedBox(width: 12),
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('PneumoScan',
                        style: TextStyle(
                          color: _textPri, fontSize: 16,
                          fontWeight: FontWeight.w700, letterSpacing: -0.3,
                        )),
                    const Text('AI Diagnostic Assistant',
                        style: TextStyle(color: _textSec, fontSize: 11)),
                  ],
                ),
                const Spacer(),
                _StatusDot(label: 'API Ready', color: _success),
                const SizedBox(width: 16),
                _StatusDot(label: 'Model v2.0', color: _cyan),
                if (showReset) ...[
                  const SizedBox(width: 20),
                  _GhostButton(
                    label: 'New Scan',
                    icon: Icons.refresh_rounded,
                    onTap: onReset,
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Hero Header (shown when no image loaded)
// ══════════════════════════════════════════════════════════════════════════════
class _HeroHeader extends StatelessWidget {
  final Animation<double> pulseAnim;
  const _HeroHeader({required this.pulseAnim});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        AnimatedBuilder(
          animation: pulseAnim,
          builder: (_, __) => Container(
            width: 80, height: 80,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(colors: [
                _accent.withOpacity(0.15 + pulseAnim.value * 0.1),
                Colors.transparent,
              ]),
            ),
            child: Center(
              child: Container(
                width: 56, height: 56,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: const LinearGradient(
                    begin: Alignment.topLeft, end: Alignment.bottomRight,
                    colors: [_accent, _cyan],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: _accent.withOpacity(0.5 + pulseAnim.value * 0.2),
                      blurRadius: 24, spreadRadius: 2,
                    ),
                  ],
                ),
                child: const Icon(Icons.document_scanner_rounded,
                    color: Colors.white, size: 28),
              ),
            ),
          ),
        ),
        const SizedBox(height: 20),
        ShaderMask(
          shaderCallback: (r) => const LinearGradient(
            colors: [_textPri, _cyan],
          ).createShader(r),
          child: const Text(
            'AI Pneumonia Detection',
            style: TextStyle(
              color: Colors.white, fontSize: 32,
              fontWeight: FontWeight.w800, letterSpacing: -0.8,
            ),
          ),
        ),
        const SizedBox(height: 10),
        const Text(
          'Upload a chest X-ray for instant AI-powered analysis\nwith GradCAM visual explainability',
          textAlign: TextAlign.center,
          style: TextStyle(color: _textSec, fontSize: 15, height: 1.6),
        ),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Stats Row
// ══════════════════════════════════════════════════════════════════════════════
class _StatsRow extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 16, runSpacing: 16,
      alignment: WrapAlignment.center,
      children: const [
        _StatCard(value: '96.2%', label: 'Validation Accuracy', icon: Icons.analytics_outlined),
        _StatCard(value: '<1s',   label: 'Inference Time',      icon: Icons.bolt_outlined),
        _StatCard(value: 'Local', label: 'Private Processing',  icon: Icons.lock_outline_rounded),
        _StatCard(value: 'GradCAM', label: 'Explainability',   icon: Icons.visibility_outlined),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String value, label;
  final IconData icon;
  const _StatCard({required this.value, required this.label, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 180,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _bg1,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _border),
      ),
      child: Column(
        children: [
          Icon(icon, color: _accent, size: 22),
          const SizedBox(height: 10),
          Text(value,
              style: const TextStyle(
                color: _textPri, fontSize: 20, fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          Text(label,
              textAlign: TextAlign.center,
              style: const TextStyle(color: _textSec, fontSize: 11)),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Layouts
// ══════════════════════════════════════════════════════════════════════════════
class _WideLayout extends StatelessWidget {
  final Uint8List? imageBytes;
  final String?    label;
  final Uint8List? heatmapBytes;
  final bool       loading;
  final String?    error;
  final Animation<double> fadeAnim;
  final AnimationController scanAnim;
  final Animation<double>  pulseAnim;
  final VoidCallback onPick;

  const _WideLayout({
    required this.imageBytes, required this.label,
    required this.heatmapBytes, required this.loading,
    required this.error, required this.fadeAnim,
    required this.scanAnim, required this.pulseAnim,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: _UploadPanel(
            imageBytes: imageBytes, loading: loading,
            scanAnim: scanAnim, onPick: onPick,
          ),
        ),
        const SizedBox(width: 20),
        Expanded(
          child: _ResultPanel(
            label: label, heatmapBytes: heatmapBytes,
            loading: loading, error: error,
            fadeAnim: fadeAnim, pulseAnim: pulseAnim,
          ),
        ),
      ],
    );
  }
}

class _NarrowLayout extends StatelessWidget {
  final Uint8List? imageBytes;
  final String?    label;
  final Uint8List? heatmapBytes;
  final bool       loading;
  final String?    error;
  final Animation<double> fadeAnim;
  final AnimationController scanAnim;
  final Animation<double>  pulseAnim;
  final VoidCallback onPick;

  const _NarrowLayout({
    required this.imageBytes, required this.label,
    required this.heatmapBytes, required this.loading,
    required this.error, required this.fadeAnim,
    required this.scanAnim, required this.pulseAnim,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _UploadPanel(
          imageBytes: imageBytes, loading: loading,
          scanAnim: scanAnim, onPick: onPick,
        ),
        const SizedBox(height: 20),
        _ResultPanel(
          label: label, heatmapBytes: heatmapBytes,
          loading: loading, error: error,
          fadeAnim: fadeAnim, pulseAnim: pulseAnim,
        ),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Upload Panel
// ══════════════════════════════════════════════════════════════════════════════
class _UploadPanel extends StatelessWidget {
  final Uint8List?          imageBytes;
  final bool                loading;
  final AnimationController scanAnim;
  final VoidCallback        onPick;

  const _UploadPanel({
    required this.imageBytes, required this.loading,
    required this.scanAnim, required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _SectionHeader(title: 'X-RAY INPUT', subtitle: 'Upload chest radiograph'),
        const SizedBox(height: 12),

        // Image well
        GestureDetector(
          onTap: loading ? null : onPick,
          child: _GlassCard(
            height: 340,
            child: loading
                ? _ScanAnimation(ctrl: scanAnim, imageBytes: imageBytes)
                : imageBytes == null
                    ? _DropZone(onTap: onPick)
                    : _XRayPreview(bytes: imageBytes!),
          ),
        ),

        const SizedBox(height: 12),

        // Upload button
        _PrimaryButton(
          label: loading ? 'Analysing…' : 'Upload X-ray Image',
          icon: loading ? null : Icons.upload_file_rounded,
          onTap: loading ? null : onPick,
          loading: loading,
        ),

        const SizedBox(height: 10),
        const _PrivacyNote(),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Result Panel
// ══════════════════════════════════════════════════════════════════════════════
class _ResultPanel extends StatelessWidget {
  final String?    label;
  final Uint8List? heatmapBytes;
  final bool       loading;
  final String?    error;
  final Animation<double> fadeAnim;
  final Animation<double> pulseAnim;

  const _ResultPanel({
    required this.label, required this.heatmapBytes,
    required this.loading, required this.error,
    required this.fadeAnim, required this.pulseAnim,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _SectionHeader(title: 'DIAGNOSIS', subtitle: 'AI analysis output'),
        const SizedBox(height: 12),

        if (error != null)
          _ErrorCard(message: error!)
        else if (loading)
          _GlassCard(
            height: 340,
            child: _LoadingState(pulseAnim: pulseAnim),
          )
        else if (label == null)
          _GlassCard(height: 340, child: const _EmptyState())
        else
          FadeTransition(
            opacity: fadeAnim,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _DiagnosisCard(label: label!),
                const SizedBox(height: 14),
                if (heatmapBytes != null)
                  _HeatmapCard(bytes: heatmapBytes!),
              ],
            ),
          ),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Scan animation overlay
// ══════════════════════════════════════════════════════════════════════════════
class _ScanAnimation extends StatelessWidget {
  final AnimationController ctrl;
  final Uint8List?           imageBytes;
  const _ScanAnimation({required this.ctrl, required this.imageBytes});

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        if (imageBytes != null)
          Image.memory(imageBytes!, fit: BoxFit.contain,
              color: Colors.black38, colorBlendMode: BlendMode.darken),
        AnimatedBuilder(
          animation: ctrl,
          builder: (_, __) {
            return CustomPaint(
              painter: _ScanLinePainter(ctrl.value),
            );
          },
        ),
        Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 48, height: 48,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _accent.withOpacity(0.15),
                  border: Border.all(color: _accent.withOpacity(0.4)),
                ),
                child: const Padding(
                  padding: EdgeInsets.all(12),
                  child: CircularProgressIndicator(
                    strokeWidth: 2, color: _accent,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              const Text('Scanning…',
                  style: TextStyle(
                    color: _accent, fontSize: 13, fontWeight: FontWeight.w600,
                    letterSpacing: 1.5,
                  )),
              const SizedBox(height: 4),
              const Text('Running AI inference',
                  style: TextStyle(color: _textSec, fontSize: 11)),
            ],
          ),
        ),
      ],
    );
  }
}

class _ScanLinePainter extends CustomPainter {
  final double t;
  _ScanLinePainter(this.t);

  @override
  void paint(Canvas canvas, Size size) {
    final y = size.height * t;
    final paint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.centerLeft, end: Alignment.centerRight,
        colors: [
          Colors.transparent,
          _cyan.withOpacity(0.8),
          _accent.withOpacity(0.9),
          _cyan.withOpacity(0.8),
          Colors.transparent,
        ],
      ).createShader(Rect.fromLTWH(0, y - 1, size.width, 2))
      ..strokeWidth = 2;
    canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);

    // Glow below line
    final glowPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter, end: Alignment.bottomCenter,
        colors: [_accent.withOpacity(0.12), Colors.transparent],
      ).createShader(Rect.fromLTWH(0, y, size.width, 40));
    canvas.drawRect(Rect.fromLTWH(0, y, size.width, 40), glowPaint);
  }

  @override
  bool shouldRepaint(_ScanLinePainter old) => old.t != t;
}

// ══════════════════════════════════════════════════════════════════════════════
// Diagnosis Card
// ══════════════════════════════════════════════════════════════════════════════
class _DiagnosisCard extends StatelessWidget {
  final String label;
  const _DiagnosisCard({required this.label});

  @override
  Widget build(BuildContext context) {
    final isPos   = label == 'PNEUMONIA';
    final color   = isPos ? _danger  : _success;
    final bgColor = isPos ? const Color(0xFF1A0A0A) : const Color(0xFF0A1A0A);
    final icon    = isPos ? Icons.warning_rounded   : Icons.verified_rounded;
    final title   = isPos ? 'Pneumonia Detected'    : 'No Pneumonia Found';
    final sub     = isPos
        ? 'Radiological signs of pneumonia are present. Recommend clinical correlation and physician consultation.'
        : 'No radiological signs of pneumonia detected. The chest X-ray appears within normal limits.';

    return Container(
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.35), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.12),
            blurRadius: 20, spreadRadius: 2,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
              border: Border(bottom: BorderSide(color: color.withOpacity(0.2))),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(icon, color: color, size: 20),
                ),
                const SizedBox(width: 12),
                Text(title,
                    style: TextStyle(
                      color: color, fontSize: 17, fontWeight: FontWeight.w700,
                    )),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    isPos ? 'POSITIVE' : 'NEGATIVE',
                    style: TextStyle(
                      color: color, fontSize: 10,
                      fontWeight: FontWeight.w700, letterSpacing: 1.2,
                    ),
                  ),
                ),
              ],
            ),
          ),
          // Body
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(sub,
                    style: const TextStyle(
                      color: _textSec, fontSize: 13, height: 1.6,
                    )),
                const SizedBox(height: 16),
                // Confidence-style indicator bar
                _ConfidenceBar(isPositive: isPos, color: color),
                const SizedBox(height: 12),
                if (isPos)
                  _DisclaimerChip(),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ConfidenceBar extends StatelessWidget {
  final bool isPositive;
  final Color color;
  const _ConfidenceBar({required this.isPositive, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text('Model Confidence',
                style: TextStyle(color: _textSec, fontSize: 11, letterSpacing: 0.5)),
            Text('High', style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: Container(
            height: 6,
            color: _bg2,
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: isPositive ? 0.87 : 0.91,
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(colors: [
                    color.withOpacity(0.6),
                    color,
                  ]),
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _DisclaimerChip extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1C1000),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFF59E0B).withOpacity(0.3)),
      ),
      child: const Row(
        children: [
          Icon(Icons.info_outline_rounded,
              size: 13, color: Color(0xFFF59E0B)),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              'For research use only. Not a substitute for clinical diagnosis.',
              style: TextStyle(
                color: Color(0xFFD97706), fontSize: 11, height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Heatmap Card
// ══════════════════════════════════════════════════════════════════════════════
class _HeatmapCard extends StatelessWidget {
  final Uint8List bytes;
  const _HeatmapCard({required this.bytes});

  @override
  Widget build(BuildContext context) {
    return _GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
            child: Row(
              children: [
                const Icon(Icons.visibility_outlined, size: 16, color: _cyan),
                const SizedBox(width: 8),
                const Text('GradCAM Activation Map',
                    style: TextStyle(
                      color: _textPri, fontSize: 13, fontWeight: FontWeight.w600,
                    )),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: _cyan.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: _cyan.withOpacity(0.3)),
                  ),
                  child: const Text('XAI',
                      style: TextStyle(
                        color: _cyan, fontSize: 10, fontWeight: FontWeight.w700,
                        letterSpacing: 1,
                      )),
                ),
              ],
            ),
          ),
          Container(height: 1, color: _border),

          ClipRRect(
            child: Image.memory(bytes, fit: BoxFit.contain),
          ),

          // Legend
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
            child: Row(
              children: [
                const Text('Cool', style: TextStyle(color: _textSec, fontSize: 10)),
                const SizedBox(width: 8),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: Container(
                      height: 5,
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(colors: [
                          Color(0xFF0000FF), Color(0xFF00FFFF),
                          Color(0xFF00FF00), Color(0xFFFFFF00),
                          Color(0xFFFF0000),
                        ]),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                const Text('Hot', style: TextStyle(color: _textSec, fontSize: 10)),
                const SizedBox(width: 16),
                const Text('Regions model focused on',
                    style: TextStyle(color: _textMut, fontSize: 10)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Drop Zone
// ══════════════════════════════════════════════════════════════════════════════
class _DropZone extends StatelessWidget {
  final VoidCallback onTap;
  const _DropZone({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 80, height: 80,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(colors: [
                _accent.withOpacity(0.15),
                _accent.withOpacity(0.03),
              ]),
              border: Border.all(
                color: _accent.withOpacity(0.25), width: 1.5,
                strokeAlign: BorderSide.strokeAlignOutside,
              ),
            ),
            child: const Icon(Icons.add_photo_alternate_outlined,
                size: 34, color: _accent),
          ),
          const SizedBox(height: 20),
          const Text('Drop X-ray here or click to browse',
              style: TextStyle(
                color: _textPri, fontSize: 14, fontWeight: FontWeight.w500,
              )),
          const SizedBox(height: 6),
          const Text('Supports JPG, PNG · Chest PA view recommended',
              style: TextStyle(color: _textSec, fontSize: 12)),
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: _accent.withOpacity(0.08),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _accent.withOpacity(0.2)),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.mouse_outlined, size: 13, color: _accent),
                SizedBox(width: 6),
                Text('Click to select file',
                    style: TextStyle(
                      color: _accent, fontSize: 12, fontWeight: FontWeight.w500,
                    )),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// X-Ray Preview
// ══════════════════════════════════════════════════════════════════════════════
class _XRayPreview extends StatelessWidget {
  final Uint8List bytes;
  const _XRayPreview({required this.bytes});

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.memory(bytes, fit: BoxFit.contain),
        // Corner brackets
        Positioned(top: 12, left: 12, child: _Corner()),
        Positioned(top: 12, right: 12, child: _Corner(flipH: true)),
        Positioned(bottom: 12, left: 12, child: _Corner(flipV: true)),
        Positioned(bottom: 12, right: 12, child: _Corner(flipH: true, flipV: true)),
        // Change hint
        Positioned(
          bottom: 0, left: 0, right: 0,
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 8),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.bottomCenter, end: Alignment.topCenter,
                colors: [Colors.black.withOpacity(0.6), Colors.transparent],
              ),
            ),
            child: const Center(
              child: Text('Tap to change image',
                  style: TextStyle(color: Colors.white60, fontSize: 11)),
            ),
          ),
        ),
      ],
    );
  }
}

class _Corner extends StatelessWidget {
  final bool flipH, flipV;
  const _Corner({this.flipH = false, this.flipV = false});

  @override
  Widget build(BuildContext context) {
    return Transform.scale(
      scaleX: flipH ? -1 : 1,
      scaleY: flipV ? -1 : 1,
      child: CustomPaint(size: const Size(16, 16), painter: _CornerPainter()),
    );
  }
}

class _CornerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = _cyan.withOpacity(0.8)
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(const Offset(0, 0), Offset(size.width, 0), p);
    canvas.drawLine(const Offset(0, 0), Offset(0, size.height), p);
  }

  @override
  bool shouldRepaint(_) => false;
}

// ══════════════════════════════════════════════════════════════════════════════
// Loading / Empty states
// ══════════════════════════════════════════════════════════════════════════════
class _LoadingState extends StatelessWidget {
  final Animation<double> pulseAnim;
  const _LoadingState({required this.pulseAnim});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: AnimatedBuilder(
        animation: pulseAnim,
        builder: (_, __) => Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 56, height: 56,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _accent.withOpacity(0.08 + pulseAnim.value * 0.06),
              ),
              child: const Padding(
                padding: EdgeInsets.all(14),
                child: CircularProgressIndicator(strokeWidth: 2, color: _accent),
              ),
            ),
            const SizedBox(height: 16),
            const Text('Processing X-ray…',
                style: TextStyle(color: _textPri, fontSize: 14, fontWeight: FontWeight.w500)),
            const SizedBox(height: 6),
            const Text('Running inference + GradCAM',
                style: TextStyle(color: _textSec, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 52, height: 52,
            decoration: BoxDecoration(
              color: _textMut.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.analytics_outlined, color: _textMut, size: 26),
          ),
          const SizedBox(height: 14),
          const Text('No analysis yet',
              style: TextStyle(color: _textSec, fontSize: 14, fontWeight: FontWeight.w500)),
          const SizedBox(height: 6),
          const Text('Upload an X-ray to see the diagnosis',
              style: TextStyle(color: _textMut, fontSize: 12)),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Error Card
// ══════════════════════════════════════════════════════════════════════════════
class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFF1A0808),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _danger.withOpacity(0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: _danger.withOpacity(0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(Icons.wifi_off_rounded, color: _danger, size: 18),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Connection Error',
                    style: TextStyle(
                      color: _danger, fontSize: 13, fontWeight: FontWeight.w600,
                    )),
                const SizedBox(height: 4),
                Text(message,
                    style: const TextStyle(
                      color: Color(0xFFF87171), fontSize: 12, height: 1.5,
                    )),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// Reusable primitives
// ══════════════════════════════════════════════════════════════════════════════
class _GlassCard extends StatelessWidget {
  final Widget child;
  final double? height;
  const _GlassCard({required this.child, this.height});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(
        color: _bg1,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 20, offset: const Offset(0, 8),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: child,
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title, subtitle;
  const _SectionHeader({required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(width: 3, height: 20,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                begin: Alignment.topCenter, end: Alignment.bottomCenter,
                colors: [_accent, _cyan],
              ),
              borderRadius: BorderRadius.circular(2),
            )),
        const SizedBox(width: 10),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: const TextStyle(
                  color: _textSec, fontSize: 10,
                  fontWeight: FontWeight.w700, letterSpacing: 1.5,
                )),
            Text(subtitle,
                style: const TextStyle(color: _textPri, fontSize: 13, fontWeight: FontWeight.w500)),
          ],
        ),
      ],
    );
  }
}

class _PrimaryButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final VoidCallback? onTap;
  final bool loading;
  const _PrimaryButton({
    required this.label, this.icon,
    this.onTap, this.loading = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        height: 48,
        decoration: BoxDecoration(
          gradient: onTap != null
              ? const LinearGradient(colors: [_accentLo, _accent])
              : null,
          color: onTap == null ? _bg2 : null,
          borderRadius: BorderRadius.circular(12),
          boxShadow: onTap != null
              ? [BoxShadow(
                  color: _accent.withOpacity(0.3),
                  blurRadius: 16, offset: const Offset(0, 6),
                )]
              : null,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (loading)
              const SizedBox(
                width: 16, height: 16,
                child: CircularProgressIndicator(
                  strokeWidth: 2, color: Colors.white60,
                ),
              )
            else if (icon != null)
              Icon(icon, size: 18, color: onTap != null ? Colors.white : _textSec),
            const SizedBox(width: 8),
            Text(label,
                style: TextStyle(
                  color: onTap != null ? Colors.white : _textSec,
                  fontSize: 14, fontWeight: FontWeight.w600,
                )),
          ],
        ),
      ),
    );
  }
}

class _StatusDot extends StatelessWidget {
  final String label;
  final Color color;
  const _StatusDot({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 6, height: 6,
          decoration: BoxDecoration(
            color: color, shape: BoxShape.circle,
            boxShadow: [BoxShadow(color: color.withOpacity(0.5), blurRadius: 4)],
          ),
        ),
        const SizedBox(width: 6),
        Text(label,
            style: const TextStyle(color: _textSec, fontSize: 11, fontWeight: FontWeight.w500)),
      ],
    );
  }
}

class _GhostButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback onTap;
  const _GhostButton({required this.label, required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: _bg2,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: _border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: _textSec),
            const SizedBox(width: 6),
            Text(label, style: const TextStyle(color: _textSec, fontSize: 12, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}

class _PrivacyNote extends StatelessWidget {
  const _PrivacyNote();

  @override
  Widget build(BuildContext context) {
    return const Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.lock_outline_rounded, size: 11, color: _textMut),
        SizedBox(width: 5),
        Text(
          'Processed locally · never sent to external servers',
          style: TextStyle(color: _textMut, fontSize: 11),
        ),
      ],
    );
  }
}
