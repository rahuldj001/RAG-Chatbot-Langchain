"""UI styling and client-side JavaScript theme injection.

Forces Gradio to default to light theme via URL parameter injection and defines
custom button colors and CSS styling.
"""

# Client-side JavaScript snippet injected on UI load
# Ensures the application always renders with the clean light theme
JS_LIGHT_THEME = """
function refresh() {
    const url = new URL(window.location);

    // If '__theme' parameter is not explicitly 'light', append it and reload
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

# Custom CSS rules for Gradio buttons (e.g. Slate standard buttons and Red stop/cancel buttons)
CSS = """
.btn {
    background-color: #64748B;
    color: #FFFFFF;
}

.stop_btn {
    background-color: #ff7373;
    color: #FFFFFF;
}
"""
