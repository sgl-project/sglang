import SwiftUI

struct MainTabView: View {
    @EnvironmentObject private var localization: LocalizationManager

    var body: some View {
        TabView {
            StudyView()
                .tabItem {
                    Label(localization.localized("tab_study"), systemImage: "book")
                }
            MockExamView()
                .tabItem {
                    Label(localization.localized("tab_mock_exam"), systemImage: "timer")
                }
            SettingsView()
                .tabItem {
                    Label(localization.localized("tab_settings"), systemImage: "gear")
                }
        }
    }
}
