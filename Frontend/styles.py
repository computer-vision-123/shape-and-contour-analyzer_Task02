COLORS = {
    'bg_warm': '#f4f6f9',      # Very light gray-blue
    'accent_red': '#e85a4f',   # Primary accent - warm red
    'accent_teal': '#88bdbc',  # Secondary accent - soft teal
    'primary_dark': '#29648a', # Dark blue
    'white': '#ffffff',        # Pure white cards
    'text_dark': '#2d3436',    # Almost black
    'text_light': '#636e72',   # Soft gray
    'success': '#88bdbc',      
    'error': '#e85a4f'
}

def get_base_styles():
    """Return base stylesheet for all widgets"""
    return f"""
        QWidget {{
            background-color: {COLORS['bg_warm']};
            color: {COLORS['text_dark']};
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 12px;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {COLORS['accent_teal']};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
            background-color: {COLORS['white']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            color: {COLORS['primary_dark']};
        }}
        
        QPushButton {{
            background-color: {COLORS['accent_teal']};
            color: {COLORS['primary_dark']};
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 600;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: {COLORS['accent_red']};
            color: white;
        }}
        QPushButton:pressed {{
            background-color: {COLORS['primary_dark']};
            color: {COLORS['white']};
        }}
        QPushButton:disabled {{
            background-color: #bdc3c7;
            color: #7f8c8d;
        }}
        
        QSlider::groove:horizontal {{
            border: none;
            height: 4px;
            background: {COLORS['accent_teal']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS['accent_red']};
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {COLORS['primary_dark']};
            transform: scale(1.1);
        }}
        
        QLabel {{
            color: {COLORS['text_dark']};
        }}
        
        QDoubleSpinBox, QSpinBox {{
            border: 1px solid {COLORS['accent_teal']};
            border-radius: 4px;
            padding: 4px 8px;
            background-color: {COLORS['white']};
            selection-background-color: {COLORS['accent_red']};
        }}
        
        QTextEdit {{
            border: 1px solid {COLORS['accent_teal']};
            border-radius: 4px;
            background-color: {COLORS['white']};
            selection-background-color: {COLORS['accent_red']};
        }}
        
        QTableWidget {{
            border: 1px solid {COLORS['accent_teal']};
            border-radius: 6px;
            background-color: {COLORS['white']};
            alternate-background-color: {COLORS['bg_warm']};
            gridline-color: {COLORS['accent_teal']};
        }}
        QTableWidget::item {{
            padding: 8px;
        }}
        QHeaderView::section {{
            background-color: {COLORS['primary_dark']};
            color: {COLORS['white']};
            padding: 8px;
            border: none;
            font-weight: 600;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {COLORS['accent_teal']};
            border-radius: 8px;
            background-color: {COLORS['white']};
        }}
        QTabBar::tab {{
            background-color: {COLORS['bg_warm']};
            color: {COLORS['text_dark']};
            padding: 8px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: 500;
        }}
        QTabBar::tab:selected {{
            background-color: {COLORS['white']};
            color: {COLORS['accent_red']};
            border-bottom: 2px solid {COLORS['accent_red']};
        }}
        QTabBar::tab:hover {{
            background-color: {COLORS['accent_teal']};
            color: {COLORS['primary_dark']};
        }}
        
        QScrollBar:vertical {{
            border: none;
            background: {COLORS['bg_warm']};
            width: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS['accent_teal']};
            border-radius: 5px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLORS['accent_red']};
        }}
        
        QStatusBar {{
            background-color: {COLORS['white']};
            color: {COLORS['text_dark']};
            border-top: 1px solid {COLORS['accent_teal']};
        }}
    """

def get_card_style():
    """Style for card-like containers"""
    return f"""
        background-color: {COLORS['white']};
        border-radius: 10px;
        padding: 12px;
        border: 1px solid {COLORS['accent_teal']};
    """

def get_title_style():
    """Style for section titles"""
    return f"""
        font-size: 18px;
        font-weight: bold;
        color: {COLORS['primary_dark']};
        padding: 8px;
        background-color: {COLORS['white']};
        border-radius: 8px;
        border-left: 4px solid {COLORS['accent_red']};
    """

def get_hint_style():
    """Style for hint/info text"""
    return f"""
        color: {COLORS['text_light']};
        font-size: 11px;
        padding: 4px;
        font-style: italic;
    """