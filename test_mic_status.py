import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt
import tellur as t
app = QApplication([])
overlay = t.Overlay(monitor_microphone=False)
image = QPixmap(360, 92)
image.fill(QColor('#20242c'))
painter = QPainter(image)
for index, state in enumerate((True, False, None)):
    overlay._set_mic_status(state, 'Microphone (KT USB Audio)')
    overlay.set_engine('local')
    assert overlay._content_height() == t.OVERLAY_HEIGHT
    assert overlay.height() == t.OVERLAY_HEIGHT + t.OVERLAY_BADGE_HEIGHT + overlay.MIC_STATUS_HEIGHT
    assert ('muted' if state is True else 'unmuted' if state is False else 'unknown') in overlay.accessibleName()
    painter.drawPixmap(index * 120 + 7, 14, overlay.grab())
painter.end()
image.save(str(Path(__file__).with_name('mic-status-preview.png')))
overlay.set_engine('pending')
assert overlay._content_height() == t.OVERLAY_HEIGHT
assert overlay.height() == t.OVERLAY_HEIGHT + overlay.MIC_STATUS_HEIGHT
print('PASS muted / unmuted / unknown UI states; engine badge layout preserved')
print('Live communications mic:', t.read_communications_mic_mute())
