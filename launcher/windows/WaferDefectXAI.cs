// 노트북용 실행 파일: 보드의 검사 화면을 주소창 없는 전용 창(Edge 앱 모드)으로 연다.
// 빌드는 같은 폴더의 build.cmd (윈도우 기본 C# 컴파일러, 설치 불필요).
using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Threading;
using System.Windows.Forms;

static class WaferDefectXAI
{
    const string DefaultUrl = "http://192.168.137.2:5000/";   // 노트북 인터넷 공유(랜선)일 때 보드 주소

    [STAThread]
    static void Main()
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string conf = Path.Combine(dir, "server.txt");           // 보드 주소가 바뀌면 이 파일 첫 줄에 적는다
        string url = File.Exists(conf) ? File.ReadAllText(conf).Trim() : DefaultUrl;
        if (url.Length == 0) url = DefaultUrl;

        if (!WaitServer(url, 20))
        {
            MessageBox.Show("검사 서버에 연결할 수 없습니다.\n\n주소: " + url +
                            "\n\n보드 전원과 랜선을 확인한 뒤 다시 실행하세요.\n(주소가 다르면 같은 폴더의 server.txt 에 적습니다.)",
                            "Wafer Defect XAI", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        string profile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "WaferDefectXAI");
        string args = "--app=" + url + " --start-maximized --no-first-run --user-data-dir=\"" + profile + "\"";
        foreach (string browser in new[] { "msedge.exe", "chrome.exe" })
        {
            try { Process.Start(new ProcessStartInfo(browser, args) { UseShellExecute = true }); return; }
            catch (System.ComponentModel.Win32Exception) { }
        }
        MessageBox.Show("Edge 또는 Chrome 을 찾을 수 없습니다.", "Wafer Defect XAI", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    // 보드를 막 켰을 때 서버가 뜰 때까지 기다린다 (1초 간격, 최대 tries 번)
    static bool WaitServer(string url, int tries)
    {
        for (int i = 0; i < tries; i++)
        {
            try
            {
                var req = (HttpWebRequest)WebRequest.Create(url);
                req.Timeout = 1500;
                using (req.GetResponse()) return true;
            }
            catch (WebException e) { if (e.Response != null) return true; }   // 응답이 오면(오류 코드라도) 서버는 살아 있다
            Thread.Sleep(1000);
        }
        return false;
    }
}
