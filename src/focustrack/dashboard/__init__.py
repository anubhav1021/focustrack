"""Layer 5 - the Streamlit dashboard.

Run it with ``focustrack dashboard``. The page itself lives in ``app.py``;
``data.py`` and ``theme.py`` hold the loading and charting logic so they can be
tested without Streamlit.
"""

from focustrack.dashboard.data import DashboardData, load_dashboard_data

__all__ = ["DashboardData", "load_dashboard_data"]
