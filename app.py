import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. 页面配置与字体处理
# ==========================================
st.set_page_config(page_title="原版定投策略回测", layout="wide")
st.title("📈 复杂策略定投回测 (PB+MA120逃顶版)")

# 解决云端中文显示
def configure_plots():
    plt.rcParams['axes.unicode_minus'] = False
    fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'sans-serif']
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            # 验证字体是否真的可用
            from matplotlib.font_manager import findfont, FontProperties
            if findfont(FontProperties(family=[font])):
                break
        except:
            continue
configure_plots()

# ==========================================
# 2. 数据加载 (适配 Streamlit)
# ==========================================
@st.cache_data
def load_data_dict(uploaded_file):
    """
    完全复用你原代码的数据解析逻辑，将所有指数数据解析为字典
    """
    indices_data = {}
    try:
        df_raw = pd.read_csv(uploaded_file, header=None)
        names_row = 3
        start_data_row = 5
        
        close_names = df_raw.iloc[names_row, 0:37].values; close_names[0] = 'date'
        pb_names = df_raw.iloc[names_row, 38:75].values; pb_names[0] = 'date'
        
        df_close = df_raw.iloc[start_data_row:, 0:37].copy(); df_close.columns = close_names
        df_close['date'] = pd.to_datetime(df_close['date'], errors='coerce')
        df_close.set_index('date', inplace=True)
        
        df_pb = df_raw.iloc[start_data_row:, 38:75].copy(); df_pb.columns = pb_names
        df_pb['date'] = pd.to_datetime(df_pb['date'], errors='coerce')
        df_pb.set_index('date', inplace=True)
        
        valid_tickers = [t for t in close_names[1:] if isinstance(t, str)]
        for t in valid_tickers:
            s_close = pd.to_numeric(df_close[t], errors='coerce')
            s_pb = pd.to_numeric(df_pb[t], errors='coerce')
            df_t = pd.DataFrame({'close': s_close, 'pb': s_pb})
            df_t.dropna(inplace=True)
            df_t.sort_index(inplace=True)
            # 原逻辑：数据量大于1250才处理（因为要计算rolling window）
            if len(df_t) > 1250: 
                indices_data[t] = df_t
        
        return indices_data
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return {}

# ==========================================
# 3. 核心策略逻辑 (完全保留 BacktestTool.run)
# ==========================================
def run_strategy(df_origin, tp_config, mtop_threshold, initial_capital=1000000, bond_yield=0.03):
    # 复制数据防止修改原件
    df = df_origin.copy()
    
    # --- 原代码常量 ---
    WINDOW_SIZE = 1250
    MA_EXIT_WINDOW = 120
    INVEST_PERIOD_DAYS = 500
    MA_EXIT_BUFFER_PCT = 0.03
    BASE_POSITION_PCT = 0.30
    FEE_RATE = 0.0001
    
    # --- 指标计算 ---
    df['ma120'] = df['close'].rolling(window=MA_EXIT_WINDOW).mean()
    df['pb_min'] = df['pb'].rolling(window=WINDOW_SIZE).quantile(0.05)
    df['pb_max'] = df['pb'].rolling(window=WINDOW_SIZE).quantile(0.95)
    range_val = df['pb_max'] - df['pb_min']
    # 避免除以0
    df['pb_score'] = np.where(range_val == 0, 0, (df['pb'] - df['pb_min']) / range_val)
    
    # --- 初始化状态 ---
    cash = initial_capital
    shares = 0.0
    avg_cost = 0.0
    history = []
    events = []
    tp_triggered = [False] * len(tp_config)
    breakdown_count = 0
    recovered_flag = False
    daily_yield_rate = (1 + bond_yield) ** (1/252) - 1
    
    # 确定遍历起始点
    start_idx = max(WINDOW_SIZE, MA_EXIT_WINDOW)
    if start_idx >= len(df):
        return None, []

    # --- 逐日回测循环 ---
    for i in range(start_idx, len(df)):
        date = df.index[i]
        price = df['close'].iloc[i]
        pb_score = df['pb_score'].iloc[i]
        ma120 = df['ma120'].iloc[i]
        
        if pd.isna(pb_score) or pd.isna(ma120): continue
        
        # 现金理财收益
        if cash > 0: cash *= (1 + daily_yield_rate)
        
        equity = cash + shares * price
        current_pos_pct = (shares * price) / equity if equity > 0 else 0
        
        # === 1. 底仓买入逻辑 ===
        if pb_score < 0.20 and current_pos_pct < 0.05:
            target_spend = equity * BASE_POSITION_PCT
            if target_spend <= cash:
                buy_val = target_spend
                fee = buy_val * FEE_RATE
                new_shares = (buy_val - fee) / price
                
                if shares > 0: avg_cost = (shares * avg_cost + buy_val) / (shares + new_shares)
                else: avg_cost = buy_val / new_shares
                
                shares += new_shares
                cash -= buy_val
                events.append({'date': date, 'price': price, 'type': '底仓买入', 'color': 'green', 'marker': '^'})
                # 更新一下equity
                equity = cash + shares * price

        # === 2. 每日定投逻辑 (基于PB分数的动态定投) ===
        daily_invest = equity / INVEST_PERIOD_DAYS
        buy_val = 0
        if pb_score < 0.00: buy_val = daily_invest * 2.0
        elif pb_score < 0.10: buy_val = daily_invest * 1.0
        elif pb_score < 0.20: buy_val = daily_invest * 0.5
        
        if buy_val > 0 and buy_val <= cash:
            fee = buy_val * FEE_RATE
            new_shares = (buy_val - fee) / price
            
            if shares > 0: avg_cost = (shares * avg_cost + buy_val) / (shares + new_shares)
            else: avg_cost = buy_val / new_shares
            
            shares += new_shares
            cash -= buy_val

        # === 3. 卖出逻辑 (止盈 + MA120逃顶) ===
        if shares > 0:
            # 止盈检查
            ret = (price / avg_cost) - 1
            for j, level in enumerate(tp_config):
                if not tp_triggered[j] and ret >= level['return']:
                    sell_shares = shares * level['sell_pct']
                    val_sold = sell_shares * price
                    fee = val_sold * FEE_RATE
                    cash += val_sold - fee
                    shares -= sell_shares
                    tp_triggered[j] = True
                    events.append({'date': date, 'price': price, 'type': f'止盈 {int(level["return"]*100)}%', 'color': 'purple', 'marker': '*'})
            
            # 逃顶检查 (仅当高估时触发)
            if pb_score > mtop_threshold:
                is_below_limit = price < ma120 * (1 - MA_EXIT_BUFFER_PCT)
                is_above_ma = price > ma120
                
                if breakdown_count == 0:
                    if is_below_limit:
                        breakdown_count = 1
                        recovered_flag = False
                        events.append({'date': date, 'price': price, 'type': '预警', 'color': 'orange', 'marker': 'x'})
                elif breakdown_count == 1:
                    if is_above_ma:
                        recovered_flag = True
                    elif is_below_limit and recovered_flag:
                        # 确认跌破，清仓
                        val_sold = shares * price
                        fee = val_sold * FEE_RATE
                        cash += val_sold - fee
                        shares = 0
                        tp_triggered = [False] * len(tp_config) # 重置止盈
                        breakdown_count = 0
                        recovered_flag = False
                        avg_cost = 0
                        events.append({'date': date, 'price': price, 'type': '清仓', 'color': 'red', 'marker': 'v'})
            else:
                breakdown_count = 0
                recovered_flag = False
        
        # 记录
        stock_val = shares * price
        history.append({
            'date': date, 
            'nav': cash + stock_val, 
            'cash': cash, 
            'stock': stock_val, 
            'close': price
        })
        
    return pd.DataFrame(history).set_index('date'), events

# ==========================================
# 4. Streamlit 界面交互
# ==========================================

# 侧边栏：参数区
st.sidebar.header("⚙️ 策略参数")
default_csv = "申万行业及宽基指数.csv"
uploaded_file = st.sidebar.file_uploader("上传数据", type=['csv'])

# 尝试加载默认数据
if not uploaded_file:
    import os
    if os.path.exists(default_csv):
        uploaded_file = default_csv
        st.sidebar.info("使用默认内置数据")

if uploaded_file:
    # 1. 加载数据
    indices_data = load_data_dict(uploaded_file)
    
    if indices_data:
        idx_names = list(indices_data.keys())
        # 默认选中“创业板指”
        default_idx = idx_names.index('创业板指') if '创业板指' in idx_names else 0
        target_index = st.sidebar.selectbox("回测标的", idx_names, index=default_idx)
        
        # 2. 参数设置
        bond_yield = st.sidebar.number_input("现金/债基年化收益率", value=0.03, step=0.01, format="%.2f")
        mtop_threshold = st.sidebar.slider("逃顶 PB 分数阈值 (MTOP)", 0.0, 1.0, 0.30, 0.05)
        
        st.sidebar.subheader("分批止盈配置")
        # 简单的动态列表模拟
        tp_levels = st.sidebar.number_input("止盈级数", 1, 5, 3)
        tp_config = []
        for i in range(tp_levels):
            c1, c2 = st.sidebar.columns(2)
            # 默认值参考你的代码：30%/20%, 60%/30%, 100%/50%
            def_ret = [30.0, 60.0, 100.0, 150.0, 200.0]
            def_sell = [20.0, 30.0, 50.0, 100.0, 100.0]
            
            r = c1.number_input(f"Level {i+1} 收益(%)", value=def_ret[i] if i<5 else 50.0, key=f"r{i}")
            s = c2.number_input(f"Level {i+1} 卖出(%)", value=def_sell[i] if i<5 else 50.0, key=f"s{i}")
            tp_config.append({'return': r/100, 'sell_pct': s/100})
            
        # 3. 运行回测
        if st.button("🚀 开始回测", type="primary"):
            st.divider()
            with st.spinner("策略回测计算中..."):
                res, events = run_strategy(
                    indices_data[target_index], 
                    tp_config, 
                    mtop_threshold, 
                    initial_capital=1000000, 
                    bond_yield=bond_yield
                )
            
            if res is not None and not res.empty:
                # 4. 结果计算
                final_nav = res['nav'].iloc[-1]
                initial_nav = 1000000
                total_ret = (final_nav / initial_nav - 1) * 100
                
                # 指标展示
                c1, c2, c3 = st.columns(3)
                c1.metric("初始资金", "1,000,000")
                c2.metric("最终净值", f"{final_nav:,.0f}")
                c3.metric("总收益率", f"{total_ret:.2f}%", delta=f"{total_ret:.2f}%")
                
                # 5. 绘图 (严格按照你的 plot 函数复刻)
                st.subheader("资产与净值走势")
                
                # 创建画布
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # 图1: 资产配置 (堆叠图)
                ax1.stackplot(res.index, res['stock'], res['cash'], 
                              labels=['股票市值', '现金/债基'], colors=['#d62728', '#95a5a6'], alpha=0.8)
                ax1.set_title(f"{target_index} - 资产配置", fontsize=14, fontweight='bold')
                ax1.set_ylabel('资产金额')
                ax1.legend(loc='upper left', framealpha=0.8, fancybox=True) 
                ax1.grid(True, alpha=0.3)
                
                # 图2: 净值与信号
                ax2.plot(res.index, res['nav'], color='#d62728', linewidth=2, label='策略净值', zorder=1)
                
                # 基准 (按照第一天的比例对齐)
                base_nav = res['nav'].iloc[0]
                bench_nav = res['close'] / res['close'].iloc[0] * base_nav
                ax2.plot(res.index, bench_nav, color='gray', linestyle=':', label='指数基准', zorder=1)
                
                # 绘制交易信号 (完全保留你的逻辑)
                if events:
                    evt_df = pd.DataFrame(events)
                    types = list(set([e['type'] for e in events]))
                    colors = {'底仓买入': 'green', '预警': 'orange', '清仓': 'red'}
                    markers = {'底仓买入': '^', '预警': 'x', '清仓': 'v'}
                    
                    for t in types:
                        if '止盈' in t: c = 'purple'; m = '*'
                        else: c = colors.get(t, 'blue'); m = markers.get(t, 'o')
                        
                        subset = evt_df[evt_df['type'] == t]
                        
                        # 🔥🔥🔥 你的关键修改：Y轴坐标取当时的净值(nav) 🔥🔥🔥
                        y_values = res.loc[subset['date'], 'nav']
                        
                        ax2.scatter(subset['date'], y_values, marker=m, color=c, s=80, label=t, zorder=5)

                ax2.set_title("净值增长与交易信号", fontsize=14, fontweight='bold')
                ax2.set_ylabel('净值 (元)')
                
                # 图例设置
                ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=3, framealpha=0.9, fancybox=True, shadow=True)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig) # 使用 Streamlit 的方法显示图表
                
                # 6. 显示详细数据
                with st.expander("查看详细交易记录"):
                    st.dataframe(pd.DataFrame(events))
                    st.dataframe(res)
            else:
                st.warning("该指数在选定参数下无法计算（可能数据长度不足1250天以计算PB分位点）。")
    else:
        st.error("数据文件中没有找到符合要求的指数（需大于1250天数据）。")