🚖 Taxi-RL Pro: Reinforcement Learning Simulation Dashboard

📌 Proje Açıklaması

Taxi-RL Pro, Q-Learning algoritması kullanılarak geliştirilmiş bir Reinforcement Learning (Pekiştirmeli Öğrenme) simülasyonudur. Bu proje, bir taksinin yolcuyu alıp hedef noktaya en verimli şekilde ulaştırmasını öğrenmesini sağlar.

Proje; eğitim süreci, performans analizi ve test simülasyonlarını hem görsel hem de metriksel olarak sunan profesyonel bir kontrol paneli içerir.

🚀 Özellikler
📊 Performans Dashboard'u
Eğitim özeti (epsilon, alpha, toplam bölüm)
Test sonuçları tablosu
📈 Grafik Analizi
Ödül yakınsaması (reward convergence)
Adım sayısı optimizasyonu (efficiency trend)
🧠 Q-Learning Algoritması
Durum-aksiyon tablosu (Q-table)
Epsilon-greedy stratejisi
🧪 Test Simülasyonu
5 farklı test turu
Gerçek zamanlı ASCII grid animasyonu
🎨 Terminal UI
Renkli ve dinamik çıktı
Canlı güncellenen grid yapısı
#----------------------------#

import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output


    def show_pro_dashboard(history_rewards, history_steps, test_results):
    clear_output(wait=True)
    
    print("\033[1;36m" + "="*65)
    print("      TAXI-RL PRO: PERFORMANS VE ANALİZ KONTROL PANELİ")
    print("="*65 + "\033[0m")
    
    # Genel Eğitim Özeti (Tablo 1)
    report_data = {
        "Eğitim Metriği": ["Toplam Bölüm", "Epsilon Final", "Öğrenme Oranı (Alpha)", "En İyi Tur"],
        "Değer": [len(history_rewards), round(epsilon, 4), alpha, f"{min(history_steps)} Adım"]
    }
    print("\n\033[1m[EĞİTİM ÖZETİ]\033[0m")
    print(pd.DataFrame(report_data).to_string(index=False))
    
    # Test Sürüşü Detayları (Tablo 2)
    print("\n\033[1m[5 TUR TEST SONUÇLARI]\033[0m")
    test_df = pd.DataFrame(test_results)
    print(test_df.to_string(index=False))
    
    print("\n" + "-"*65)

    # Profesyonel Grafikler
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Grafik 1: Ödül Yakınsaması
    smoothed_r = pd.Series(history_rewards).rolling(window=100).mean()
    ax1.plot(history_rewards, color='limegreen', alpha=0.15, label='Ham Veri')
    ax1.plot(smoothed_r, color='darkgreen', linewidth=2.5, label='Eğilim (100 Bölüm Ort.)')
    ax1.set_title("Öğrenme Stabilitesi: Kümülatif Ödül", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Bölüm"); ax1.set_ylabel("Toplam Ödül")
    ax1.legend()

    # Grafik 2: Operasyonel Hız
    smoothed_s = pd.Series(history_steps).rolling(window=100).mean()
    ax2.plot(history_steps, color='salmon', alpha=0.15, label='Ham Veri')
    ax2.plot(smoothed_s, color='darkred', linewidth=2.5, label='Hız Eğilimi (100 Bölüm Ort.)')
    ax2.set_title("Operasyonel Verimlilik: Adım Sayısı", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Bölüm"); ax2.set_ylabel("Adım Sayısı")
    ax2.set_ylim(0, 110)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    test_stats = []

    for tur in range(1, 6):
        state = env.reset()
        epochs, total_reward, done = 0, 0, False
    
    while not done and epochs < 50:
        action = np.argmax(q_table[state])
        next_state, reward, done = env.step(action)
        
        # ASCII Dinamik Grid
        t_row, t_col = (state // (6*7*6)) % 6, (state // (7*6)) % 6
        grid = "    + - - - - - - - - +\n"
        for r in range(6):
            grid += "    | "
            for c in range(6):
                char = "."
                for i, p in enumerate(env.locs):
                    if (r,c) == p: char = "RGYBKL"[i]
                if r == t_row and c == t_col:
                    grid += f"\033[43m\033[30m{char}\033[0m"
                else: grid += char
                grid += " | " if ((r<2 and c==1) or (r>3 and c==2)) else " : "
            grid += "|\n"
        grid += "    + - - - - - - - - +"
        
        clear_output(wait=True)
        print(f"\033[1;36m[PRO TEST] TUR: {tur}/5 | ADIM: {epochs}\033[0m")
        print(grid)
        print(f"\033[1mKonum:\033[0m ({t_row},{t_col}) | \033[1mAksiyon:\033[0m {['Güney','Kuzey','Doğu','Batı','PICKUP','DROPOFF'][action]}")
        print(f"\033[1mAnlık Ödül:\033[0m {reward} | \033[1mToplam:\033[0m {total_reward + reward}")
        
        state, total_reward, epochs = next_state, total_reward + reward, epochs + 1
        time.sleep(0.2)

    if done:
        test_stats.append({"Tur": tur, "Toplam Ödül": total_reward, "Adım Sayısı": epochs, "Sonuç": "BAŞARILI"})
        time.sleep(1)

    show_pro_dashboard(history_rewards, history_steps, test_stats)


📄 Performans ve Analiz Kontrol Paneli:: https://drive.google.com/file/d/1rD071WektDtPoRhxjMWjEgwl7rIOLxS5/view

📄 Grafikler:: https://drive.google.com/file/d/1YBp0VUIQJb75jcEqHQvt0vGSxsL8u6Vd/view

📄 Simülasyon Gif:: https://drive.google.com/file/d/1lA3LVKWnbtTiDwgsZw8yK0x-RGxlc05R/view
