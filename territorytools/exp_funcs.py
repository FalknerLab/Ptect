

def plot_all_urine_xys(g_id, g_data, g_info, params):
    c, ax = params[:]
    group_n = len(g_data)
    for g in g_data:
        ax.scatter(g[:, 0], g[:, 1], color=c, s=1)
    ax.set_title('Group: ' + g_id)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    return None


def plot_block0(run_data):
    times, evt_xys = run_data[:]
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig)
    total_marks_left, total_marks_right = urine_area_over_time(run_data)
    times = np.arange(len(total_marks_left)) / (40 * 60)
    y1 = max(np.max(total_marks_left), np.max(total_marks_right))
    y1 = y1 + 0.2 * y1
    y1 = 1200
    ax0 = fig.add_subplot(gs[1, 2:])
    ax0.plot(times, total_marks_right, c=[1, 0.5, 0])
    ax0.set_ylim(0, y1)
    ax0.set_xlabel('Time (min)')
    ax0.set_ylabel('Urine Area (px)')
    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.plot(times, total_marks_left, c=[0, 0.8, 0])
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Urine Area (px)')
    ax1.set_ylim(0, y1)
    _, mask_l, mask_r = get_mask(run_data)
    mask_h, mask_w = np.shape(mask_l)
    rgb = np.zeros((mask_h, mask_w, 3))
    rgb[:, :, 1] += mask_l
    rgb[:, :, 0] += mask_r
    rgb[:, :, 1] += mask_r / 2
    ax2 = fig.add_subplot(gs[:, :2])
    ax2.imshow(np.flipud(rgb))
    plt.show()

def plot_all_urine_xys(g_id, g_data, g_info, params):
    c, ax = params[:]
    group_n = len(g_data)
    for g in g_data:
        ax.scatter(g[:, 0], g[:, 1], color=c, s=1)
    ax.set_title('Group: ' + g_id)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    return None

def plot_all_masks(g_id, g_data, g_info):
    f, axs = plt.subplots(1, 1, figsize=(10, 10))
    m0 = g_data[0]
    mask = np.zeros_like(m0)
    for g in g_data:
        mask += g
    axs.imshow(mask)
    axs.set_title(g_id)

def plot_all_series_and_mean(g_id, g_data, g_info):
    f, axs = plt.subplots(2, 1, figsize=(20, 10))
    m0 = g_data[0]
    num_f = len(m0[0])
    temp = np.zeros((num_f, len(g_data) * 2))
    c = -1
    for m in g_data:
        c += 1
        t = np.arange(num_f) / (40 * 60)
        axs[0].plot(t, m[0], c=[0.5, 0.5, 0.5])
        axs[0].plot(t, m[1], c=[0.5, 0.5, 0.5])
        temp[:, c] = m[0]
        temp[:, c + 1] = m[1]
    axs[1].plot(t, np.mean(temp, axis=1), c='r')
    plt.show()
    print('yes')

def run_cc(g_id, g_data, g_info):
    for g, g_i in zip(g_data, g_info):
        u_r, u_i = urine_area_over_time(g)
        corr = correlate(u_r, u_i)
        lags = correlation_lags(len(u_r), len(u_i))
        corr /= np.max(corr)
        plt.figure()
        plt.plot(lags / (40 * 60), corr)
        f_name = g_i['Resident'] + ' vs ' + g_i['Intruder'] + ' Day ' + g_i['Day']
        plt.title(f_name)
        plt.show()