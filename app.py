import streamlit as st
import pandas as pd
import requests
import io
import plotly.graph_objects as go
from datetime import datetime
import re
from bs4 import BeautifulSoup
import yfinance as yf

# --- Configuration ---
st.set_page_config(
    page_title="CFTC COT Analyzer - Soybeans",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
# URL Template for Disaggregated Futures-and-Options Combined Reports
BASE_URL = "https://www.cftc.gov/files/dea/history/com_disagg_txt_{}.zip"

COLUMN_MAP = {
    'Report_Date_as_YYYY-MM-DD': 'Date',
    'Market_and_Exchange_Names': 'Market',
    'M_Money_Positions_Long_All': 'Long',
    'M_Money_Positions_Short_All': 'Short'
}

TARGET_MARKET_SOY = "SOYBEANS - CHICAGO BOARD OF TRADE"
TARGET_MARKET_CORN = "CORN - CHICAGO BOARD OF TRADE"
TARGET_MARKET_SOY_OIL = "SOYBEAN OIL - CHICAGO BOARD OF TRADE"
TARGET_MARKET_SOY_MEAL = "SOYBEAN MEAL - CHICAGO BOARD OF TRADE"

MARKET_TO_TICKER = {
    TARGET_MARKET_SOY: "ZS=F",
    TARGET_MARKET_CORN: "ZC=F",
    TARGET_MARKET_SOY_OIL: "ZL=F",
    TARGET_MARKET_SOY_MEAL: "ZM=F"
}

# --- Data Loading (CFTC) ---
@st.cache_data
def load_data(years):
    """Downloads and parses CFTC COT data for selected years."""
    all_data = []
    
    for year in years:
        url = BASE_URL.format(year)
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            
            with io.BytesIO(response.content) as zip_buffer:
                import zipfile
                with zipfile.ZipFile(zip_buffer) as z:
                    file_name = z.namelist()[0]
                    with z.open(file_name) as f:
                        df = pd.read_csv(f, low_memory=False)
                        df.columns = df.columns.str.strip()
                        
                        if 'Market_and_Exchange_Names' in df.columns:
                             # Filter for target markets
                             target_markets = [
                                 TARGET_MARKET_SOY, 
                                 TARGET_MARKET_CORN,
                                 TARGET_MARKET_SOY_OIL,
                                 TARGET_MARKET_SOY_MEAL
                             ]
                             df = df[df['Market_and_Exchange_Names'].str.strip().isin(target_markets)].copy()
                        
                        all_data.append(df)
                        
        except Exception as e:
            st.error(f"Error loading data for {year}: {e}")
            continue
            
    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# --- Processing (CFTC) ---
def process_data(df):
    if df.empty:
        return df
        
    needed_cols = list(COLUMN_MAP.keys())
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
        return pd.DataFrame()

    df = df[needed_cols].rename(columns=COLUMN_MAP)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce').fillna(0)
    df['Short'] = pd.to_numeric(df['Short'], errors='coerce').fillna(0)
    df['Net_Position'] = df['Long'] - df['Short']
    
    return df

def calculate_indicators(df, sma_window):
    if df.empty: return df
    df['SMA'] = df['Net_Position'].rolling(window=sma_window).mean()
    mean = df['Net_Position'].mean()
    std = df['Net_Position'].std()
    df['Z_Score'] = (df['Net_Position'] - mean) / std
    return df

# --- WASDE Integration (USDA XML) ---

@st.cache_data(ttl=3600)
def get_wasde_xml_urls():
    """Finds the top 2 latest WASDE XML URLs from Cornell ESMIS."""
    base_url = "https://usda.library.cornell.edu/concern/publications/3t945q76s"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(base_url, headers=headers, verify=False)
        r.raise_for_status()

        # Find all hrefs ending in .xml
        links = re.findall(r'href=["\']([^"\']+\.xml)["\']', r.text, re.IGNORECASE)
        
        candidates = []
        seen = set()
        
        for l in links:
             # Resolve relative
             full_link = l if l.startswith('http') else f"https://usda.library.cornell.edu{l}" if l.startswith('/') else f"https://usda.library.cornell.edu/{l}"
             
             if full_link not in seen:
                 candidates.append(full_link)
                 seen.add(full_link)
        
        # Filter for "wasde" in name to avoid ancillary files if any
        final_list = [c for c in candidates if 'wasde' in c.lower()]
        
        if not final_list:
             final_list = candidates
             
        # Return top 2 (Current, Previous)
        return final_list[:2]

    except Exception as e:
        st.error(f"Error searching WASDE URLs: {e}")
        return []

def extract_date_from_url(url):
    """Extracts Date Label (e.g. 'Jan/26') from URL."""
    match = re.search(r"wasde(\d{2})(\d{2})", url, re.IGNORECASE)
    if match:
        month = match.group(1)
        year = match.group(2)
        try:
            dt = datetime.strptime(f"{month} {year}", "%m %y")
            months_pt = {
                1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
                7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
            }
            return f"{months_pt[dt.month]}/{year}"
        except:
            pass
    return "Relatório"

@st.cache_data(ttl=3600)
def fetch_wasde_xml_data(url):
    """Downloads and parses a SINGLE WASDE report from XML with Anchored Scoping."""
    try:
        r = requests.get(url, verify=False)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.content, "xml")
        data = {}
        
        def extract_cell_value(tag):
            # 1. Year Groups (Prioritize)
            ygs = tag.find_all(re.compile(r"year_group", re.IGNORECASE))
            if ygs:
                last_yg = ygs[-1]
                cells = last_yg.find_all(re.compile(r"cell", re.IGNORECASE))
                if cells:
                    vals = [v for k,v in cells[-1].attrs.items() if 'value' in k]
                    if vals: return float(vals[0].replace(',', ''))
            
            # 2. Direct Cells
            cells = tag.find_all(re.compile(r"cell", re.IGNORECASE))
            if cells:
                vals = [v for k,v in cells[-1].attrs.items() if 'value' in k]
                if vals: return float(vals[0].replace(',', ''))
            return None

        def get_value_anchored(report_regex, attr_term, anchor_term=None, region_filter=None):
            """
            Finds a value by locating a Report, then identifying the correct 'Block' (sibling group)
            using an Anchor Term (e.g. 'Ethanol' implies Corn block).
            """
            reports = soup.find_all('Report', attrs={'sub_report_title': re.compile(report_regex, re.IGNORECASE)})
            if not reports: return None
            
            for report in reports:
                # 1. Identify the Target Attribute ID (e.g. attribute2)
                # If anchor provided, find which attribute group contains the anchor
                target_attr_key = None
                
                if anchor_term:
                    # Find tag containing anchor
                    anchor_tags = report.find_all(lambda tag: tag.attrs and any(anchor_term.lower() in str(v).lower() for v in tag.attrs.values()))
                    for at in anchor_tags:
                        for k, v in at.attrs.items():
                             if 'attribute' in k: 
                                 target_attr_key = k
                                 break
                        if target_attr_key: break
                
                # If no anchor or not found, try generic search or use found key
                # Search for the Target Attribute (e.g. Yield)
                candidates = report.find_all(lambda tag: tag.attrs and any(attr_term.lower() in str(v).lower() for k,v in tag.attrs.items() if 'attribute' in k))
                
                for cand in candidates:
                    # Filter by Region if needed
                    # If region_filter provided, we require it in the ancestry OR in the attribute itself
                    if region_filter:
                         # Ancestry check
                         parents = list(cand.parents)
                         found_reg = False
                         
                         # Check immediate attributes first (sometimes region implied in key?)
                         # Also check if 'World' is the default for this report and we want 'World'
                         # If we want 'World' and no region tag exists, maybe it IS world?
                         # Only assume this if strict region tagging failed.
                         
                         # Explicit check
                         for p in parents:
                             if p==report: break
                             if p.attrs and any(region_filter.lower() in str(v).lower() for v in p.attrs.values()):
                                 found_reg = True; break
                         
                         if not found_reg:
                             # Looser check for World in World Report
                             if region_filter.lower() == "world" and "world" in report.attrs.get('sub_report_title', '').lower():
                                 # Checking if this candidate is NOT nested under *another* region
                                 # e.g. parent has region='Brazil'?
                                 is_other_region = False
                                 for p in parents:
                                     if p==report: break
                                     if p.attrs and any('region' in k and region_filter.lower() not in str(v).lower() for k,v in p.attrs.items()):
                                         is_other_region = True
                                         break
                                 if not is_other_region:
                                     found_reg = True
                         
                         if not found_reg: continue

                    # Filter by Anchor Key (if identified)
                    if target_attr_key:
                        if not any(k == target_attr_key for k in cand.attrs.keys()):
                             continue

                    return extract_cell_value(cand)
            return None

        # --- Extraction ---
        
        # 1. US Corn
        # Use "Ethanol" to identify Corn block (attribute2 usually)
        data["us_corn_yield"] = get_value_anchored(r"U\.S\. Feed Grain", "Yield", anchor_term="Ethanol")
        data["us_corn_stocks"] = get_value_anchored(r"U\.S\. Feed Grain", "Ending Stocks", anchor_term="Ethanol")

        # 2. US Soy
        # Report "U.S. Soybeans". Usually only one commodity, so anchor optional.
        data["us_soy_yield"] = get_value_anchored(r"U\.S\. Soybeans", "Yield")
        data["us_soy_stocks"] = get_value_anchored(r"U\.S\. Soybeans", "Ending Stocks")
        
        # 3. US Wheat
        data["us_wheat_stocks"] = get_value_anchored(r"U\.S\. Wheat", "Ending Stocks")
        
        # 4. World Stocks
        # Find "Ending Stocks" where Region is "World" in "World Corn/Soy/Wheat" report
        data["world_corn_stocks"] = get_value_anchored(r"World Corn", "Ending Stocks", region_filter="World")
        data["world_soy_stocks"] = get_value_anchored(r"World Soybean", "Ending Stocks", region_filter="World")
        data["world_wheat_stocks"] = get_value_anchored(r"World Wheat", "Ending Stocks", region_filter="World")
        
        # 5. SA Production
        data["bra_soy_prod"] = get_value_anchored(r"World Soybean", "Production", region_filter="Brazil")
        data["bra_corn_prod"] = get_value_anchored(r"World Corn", "Production", region_filter="Brazil")
        data["arg_soy_prod"] = get_value_anchored(r"World Soybean", "Production", region_filter="Argentina")
        data["arg_corn_prod"] = get_value_anchored(r"World Corn", "Production", region_filter="Argentina")

        return data

    except Exception as e:
        return {"_error": str(e)}

def wasde_ui():
    st.header("Relatório WASDE (Comparativo Mensal)")
    
    if st.button("Atualizar Dados WASDE"):
        st.cache_data.clear()
        st.rerun()
        
    with st.spinner("Buscando relatórios XML recentes..."):
        urls = get_wasde_xml_urls()
        
    if not urls:
        st.error("Nenhum relatório XML encontrado.")
        return
        
    url_current = urls[0]
    date_current = extract_date_from_url(url_current)
    url_prev = urls[1] if len(urls) > 1 else None
    date_prev = extract_date_from_url(url_prev) if url_prev else "N/A"
    
    st.info(f"Fonte: USDA XML (via Cornell Archive). Comparando: **{date_current}** vs **{date_prev}**")
    
    col1, col2 = st.columns(2)
    with col1:
        data_curr = fetch_wasde_xml_data(url_current)
    with col2:
        data_prev = fetch_wasde_xml_data(url_prev) if url_prev else {}

    if not data_curr:
        st.error("Falha ao ler o relatório XML atual.")
        return

    # --- Market Estimates Input ---
    st.markdown("### Estimativas de Mercado (Inputs Manuais)")
    with st.expander("Inserir Dados do Mercado", expanded=True):
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### Intervalo de Mercado (Min - Max)")
            input_range_soy_stock = st.text_input("Stocks Soja EUA (Min-Max)", "300-350")
            input_range_corn_stock = st.text_input("Stocks Milho EUA (Min-Max)", "1600-1700")
            input_range_bra_soy = st.text_input("Prod. Soja Brasil (Min-Max)", "160-165")
            
        with col_m2:
            st.markdown("#### Média de Mercado")
            input_avg_soy_stock = st.number_input("Stocks Soja EUA (Média)", value=320.0)
            input_avg_corn_stock = st.number_input("Stocks Milho EUA (Média)", value=1650.0)
            input_avg_bra_soy = st.number_input("Prod. Soja Brasil (Média)", value=163.0)
            
        market_estimates = {
            "us_soy_stocks": {"avg": input_avg_soy_stock, "range": input_range_soy_stock},
            "us_corn_stocks": {"avg": input_avg_corn_stock, "range": input_range_corn_stock},
            "bra_soy_prod": {"avg": input_avg_bra_soy, "range": input_range_bra_soy}
        }

    # --- Display Logic ---
    FACTOR_SOY_YIELD = 1.1209
    FACTOR_CORN_YIELD = 1.0461
    FACTOR_SOY_WHEAT_VOL = 0.027216
    FACTOR_CORN_VOL = 0.025401
    
    # Separate lists for each section
    rows_us_stocks = []
    rows_yield = []
    rows_world_stocks = []
    rows_sa_prod = []
    
    def add_row(target_list, asset_name, key, unit_original, unit_converted, factor, mkt_key=None):
        # Values
        v_c = data_curr.get(key)
        v_p = data_prev.get(key)
        
        v_c_val = float(v_c) if (v_c is not None) else 0.0
        v_p_val = float(v_p) if (v_p is not None) else 0.0
        
        has_c = v_c is not None
        has_p = v_p is not None
        
        diff = v_c_val - v_p_val
        
        # Market Data
        mkt_avg = "-"
        mkt_range = "-"
        diff_mkt = "-"
        
        if mkt_key and mkt_key in market_estimates:
             mkt_data = market_estimates[mkt_key]
             mkt_avg_val = mkt_data["avg"]
             mkt_range = mkt_data["range"]
             mkt_avg = f"{mkt_avg_val:,.2f}"
             
             if has_c:
                 diff_val = v_c_val - mkt_avg_val
                 diff_mkt = f"{diff_val:,.2f}"

        # Row 1 Original
        target_list.append({
            "Métrica": asset_name,
            "Unidade": unit_original,
            "Mercado (Média)": mkt_avg,
            "Mercado (Range)": mkt_range,
            f"Rel. {date_prev}": f"{v_p_val:,.2f}" if has_p else "-",
            f"Rel. {date_current}": f"{v_c_val:,.2f}" if has_c else "-",
            "Var (Mensal)": f"{diff:,.2f}" if (has_c and has_p) else "-",
            "Var (vs Mkt)": diff_mkt
        })
        
        # Row 2 Converted
        if factor:
             v_c_conv = v_c_val * factor
             v_p_conv = v_p_val * factor
             diff_conv = v_c_conv - v_p_conv
             
             # Convert market as well? Usually Mkt is given in original. 
             # Let's show conversion for Reported only for now to keep table clean, 
             # or imply mkt inputs are in original units.
             
             target_list.append({
                "Métrica": "",
                "Unidade": unit_converted,
                "Mercado (Média)": "",
                "Mercado (Range)": "",
                f"Rel. {date_prev}": f"{v_p_conv:,.2f}" if has_p else "-",
                f"Rel. {date_current}": f"{v_c_conv:,.2f}" if has_c else "-",
                "Var (Mensal)": f"{diff_conv:,.2f}" if (has_c and has_p) else "-",
                "Var (vs Mkt)": ""
            })

    # Build Lists
    # Only map market keys where inputs exist
    add_row(rows_us_stocks, "TRIGO", "us_wheat_stocks", "Milhões de Bushels", "Milhões de Toneladas", FACTOR_SOY_WHEAT_VOL)
    add_row(rows_us_stocks, "MILHO", "us_corn_stocks", "Milhões de Bushels", "Milhões de Toneladas", FACTOR_CORN_VOL, "us_corn_stocks")
    add_row(rows_us_stocks, "SOJA", "us_soy_stocks", "Milhões de Bushels", "Milhões de Toneladas", FACTOR_SOY_WHEAT_VOL, "us_soy_stocks")
    
    add_row(rows_yield, "MILHO", "us_corn_yield", "Bushels por Acre", "Sacas por Hectare", FACTOR_CORN_YIELD)
    add_row(rows_yield, "SOJA", "us_soy_yield", "Bushels por Acre", "Sacas por Hectare", FACTOR_SOY_YIELD)
    
    add_row(rows_world_stocks, "TRIGO (Global)", "world_wheat_stocks", "Milhões de Toneladas", "Milhões de Toneladas", None) # Already Tonnes
    add_row(rows_world_stocks, "MILHO (Global)", "world_corn_stocks", "Milhões de Toneladas", "Milhões de Toneladas", None)
    add_row(rows_world_stocks, "SOJA (Global)", "world_soy_stocks", "Milhões de Toneladas", "Milhões de Toneladas", None)
    
    add_row(rows_sa_prod, "ARGENTINA (Soja)", "arg_soy_prod", "Milhões de Toneladas", "Milhões de Toneladas", None)
    add_row(rows_sa_prod, "ARGENTINA (Milho)", "arg_corn_prod", "Milhões de Toneladas", "Milhões de Toneladas", None)
    add_row(rows_sa_prod, "BRASIL (Soja)", "bra_soy_prod", "Milhões de Toneladas", "Milhões de Toneladas", None, "bra_soy_prod") # Added mkt key
    add_row(rows_sa_prod, "BRASIL (Milho)", "bra_corn_prod", "Milhões de Toneladas", "Milhões de Toneladas", None)

    # Rendering
    def render_section(title, rows):
        st.markdown(f"### {title}")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Nenhum dado.")
            
    render_section("Estoque Final EUA", rows_us_stocks)
    render_section("Estimativa de Produtividade (Safra Atual)", rows_yield)
    render_section("Estoque Final Global", rows_world_stocks)
    render_section("Produção América do Sul", rows_sa_prod)

@st.cache_data(ttl=86400)  # Cache for 1 day
def fetch_cbot_prices(ticker, start_str, end_str):
    """Fetches historical closing prices from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker (e.g. 'ZS=F')
        start_str: Start date as 'YYYY-MM-DD' string
        end_str: End date as 'YYYY-MM-DD' string
    """
    import warnings
    try:
        # Use a standard requests session to avoid curl_cffi dependency issues on Streamlit Cloud
        session = requests.Session()
        session.verify = False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True, session=session)
        
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            close_series = data['Close'].iloc[:, 0]  # First (and only) ticker
        else:
            close_series = data['Close']
        
        # Drop NaN values
        close_series = close_series.dropna()
        
        if close_series.empty:
            return pd.DataFrame()
        
        # Create a clean dataframe
        price_df = pd.DataFrame({
            'Date': close_series.index,
            'Close': close_series.values
        })
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        # Remove timezone info if present
        if price_df['Date'].dt.tz is not None:
            price_df['Date'] = price_df['Date'].dt.tz_localize(None)
        
        return price_df
    except Exception:
        return pd.DataFrame()

def render_cot_dashboard(raw_df, market_name, market_label, sma_period):
    """
    Renders the COT dashboard for a specific market.
    """
    if raw_df.empty:
        return

    # Filter for specific market
    if 'Market' not in raw_df.columns:
        st.error(f"Column 'Market' not found in data. Columns: {raw_df.columns.tolist()}")
        return

    df = raw_df[raw_df['Market'] == market_name].copy()
    
    if df.empty:
        # Only show warning if we expected data (i.e. not empty raw_df)
        if not raw_df.empty:
             st.warning(f"No data for {market_label}.")
        return

    # Calculate indicators specific to this market subset
    df = calculate_indicators(df, sma_period)
    
    # --- Visualization ---
    
    # 1. Main Chart: Net Position & SMA
    fig_main = go.Figure()

    # Net Position Bars
    colors = ['green' if x >= 0 else 'red' for x in df['Net_Position']]
    fig_main.add_trace(go.Bar(
        x=df['Date'], 
        y=df['Net_Position'],
        name='Net Position',
        marker_color=colors
    ))

    # SMA Line
    sma = df['SMA'].values
    sma_series = pd.Series(sma)
    diff = sma_series.diff()
    is_rising = diff > 0
    is_falling = diff <= 0
    
    mask_green = is_rising | is_rising.shift(-1).fillna(False)
    mask_red = is_falling | is_falling.shift(-1).fillna(False)
    
    sma_green = sma_series.where(mask_green, None)
    sma_red = sma_series.where(mask_red, None)
    
    fig_main.add_trace(go.Scatter(x=df['Date'], y=sma_green, name=f'SMA {sma_period} (↑)', line=dict(color='lime', width=2), mode='lines', connectgaps=False))
    fig_main.add_trace(go.Scatter(x=df['Date'], y=sma_red, name=f'SMA {sma_period} (↓)', line=dict(color='red', width=2), mode='lines', connectgaps=False))

    # --- Fetch and Plot CBOT Prices ---
    ticker = MARKET_TO_TICKER.get(market_name)
    has_prices = False
    
    if ticker:
        # Get min and max dates from COT data as strings for caching
        start_date = df['Date'].min().strftime('%Y-%m-%d')
        end_date = (df['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        price_df = fetch_cbot_prices(ticker, start_date, end_date)
        
        if not price_df.empty:
            has_prices = True
            
            # Merge to align dates for hover (optional, but good for unified hover)
            # However, plotly x unified hover handles separate traces automatically if x-axis matches.
            
            fig_main.add_trace(go.Scatter(
                x=price_df['Date'],
                y=price_df['Close'],
                name=f'Price ({ticker})',
                mode='lines',
                line=dict(color='navy', width=2.5),
                yaxis='y2' # Assign to secondary y-axis
            ))

    # Set up layout for secondary y-axis
    layout_update = dict(
        title=f"Net Position (Managed Money) & CBOT Price - {market_label}",
        xaxis_title="Date",
        hovermode="x unified",
        showlegend=False,
        yaxis=dict(
            title="Contracts",
            dtick=50000,
            side='left'
        ),
        xaxis=dict(hoverformat='%d/%m/%Y'),
        height=500
    )
    
    if has_prices:
        layout_update['yaxis2'] = dict(
            title="Price",
            side='right',
            overlaying='y',
            showgrid=False # Don't conflict with main grid lines
        )
        
    fig_main.update_layout(**layout_update)
    
    config = {'toImageButtonOptions': {'format': 'png','filename': f'cot_{market_label}','height': 1080,'width': 1920,'scale': 2},'displaylogo': False}
    st.plotly_chart(fig_main, use_container_width=True, config=config)

    # 2. Z-Score Chart
    fig_z = go.Figure()
    z_colors = ['green' if x >= 0 else 'red' for x in df['Z_Score']]
    
    fig_z.add_trace(go.Bar(
        x=df['Date'],
        y=df['Z_Score'],
        name='Z-Score',
        marker_color=z_colors
    ))
    
    fig_z.add_hline(y=2, line_dash="dash", line_color="gray")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="gray")
    fig_z.add_hline(y=0, line_color="black")
    
    fig_z.update_layout(
        title=f"Z-Score ({market_label})",
        yaxis_title="Z-Score",
        xaxis_title="Date",
        hovermode="x unified",
        showlegend=False,
        yaxis=dict(dtick=0.5),
        xaxis=dict(hoverformat='%d/%m/%Y'),
        height=400
    )
    st.plotly_chart(fig_z, use_container_width=True, config=config)

    with st.expander(f"Dados Brutos - {market_label}"):
        st.dataframe(df.sort_values('Date', ascending=False))

def main():
    st.title("CFTC COT Analyzer: Managed Money")
    
    # Create Tabs
    # Order: Soja, Oleo, Farelo, Complexo, Milho
    tab_soy, tab_oil, tab_meal, tab_complex, tab_corn = st.tabs([
        "COT Soja", 
        "COT Óleo de Soja", 
        "COT Farelo de Soja", 
        "COT Complexo Soja",
        "COT Milho"
    ])
    
    # Sidebar
    st.sidebar.header("Settings")
    current_year = datetime.now().year
    years_available = range(current_year, 2009, -1)
    selected_years = st.sidebar.multiselect("Select Years", options=years_available, default=[current_year, current_year-1])
    sma_period = st.sidebar.slider("SMA Period (Weeks)", 1, 52, 26)

    # Load Data
    raw_df_processed = pd.DataFrame()
    if selected_years:
        with st.spinner("Downloading and processing data..."):
            raw_data = load_data(selected_years)
            if not raw_data.empty:
                raw_df_processed = process_data(raw_data)

    # Helper check
    has_data = not raw_df_processed.empty

    # 1. Soja
    with tab_soy:
        if has_data:
            render_cot_dashboard(raw_df_processed, TARGET_MARKET_SOY, "Soja", sma_period)
        elif selected_years: st.warning("No data found.")
        else: st.info("Select years.")

    # 2. Oleo
    with tab_oil:
        if has_data:
            render_cot_dashboard(raw_df_processed, TARGET_MARKET_SOY_OIL, "Óleo de Soja", sma_period)
        elif selected_years: st.warning("No data found.")
        else: st.info("Select years.")

    # 3. Farelo
    with tab_meal:
        if has_data:
            render_cot_dashboard(raw_df_processed, TARGET_MARKET_SOY_MEAL, "Farelo de Soja", sma_period)
        elif selected_years: st.warning("No data found.")
        else: st.info("Select years.")

    # 4. Milho
    with tab_corn:
        if has_data:
            render_cot_dashboard(raw_df_processed, TARGET_MARKET_CORN, "Milho", sma_period)
        elif selected_years: st.warning("No data found.")
        else: st.info("Select years.")

    # 6. Complexo Soja (Overlay)
    with tab_complex:
        if has_data:
            st.header("Complexo Soja: Posição Líquida Comparada")
            
            # Extract Data
            df_soy = raw_df_processed[raw_df_processed['Market'] == TARGET_MARKET_SOY].copy()
            df_oil = raw_df_processed[raw_df_processed['Market'] == TARGET_MARKET_SOY_OIL].copy()
            df_meal = raw_df_processed[raw_df_processed['Market'] == TARGET_MARKET_SOY_MEAL].copy()
            
            fig_comp = go.Figure()
            
            # Add Traces
            # Soja (Green)
            fig_comp.add_trace(go.Scatter(
                x=df_soy['Date'], y=df_soy['Net_Position'],
                name='Soja (Grão)', mode='lines', line=dict(color='green', width=2)
            ))
            
            # Oleo (Gold/Yellow)
            fig_comp.add_trace(go.Scatter(
                x=df_oil['Date'], y=df_oil['Net_Position'],
                name='Óleo de Soja', mode='lines', line=dict(color='gold', width=2)
            ))
            
            # Farelo (Brown/Orange)
            fig_comp.add_trace(go.Scatter(
                x=df_meal['Date'], y=df_meal['Net_Position'],
                name='Farelo de Soja', mode='lines', line=dict(color='peru', width=2)
            ))
            
            fig_comp.update_layout(
                title="Posição Líquida - Complexo Soja (Contratos)",
                xaxis_title="Data",
                yaxis_title="Contratos (Net)",
                hovermode="x unified",
                xaxis=dict(hoverformat='%d/%m/%Y'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
        elif selected_years: st.warning("No data found.")
        else: st.info("Select years.")

if __name__ == "__main__":
    main()
