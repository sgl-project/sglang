import SwiftUI

@main
struct CivicsPrepApp: App {
    @StateObject private var container = AppContainer()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(container.localizationManager)
                .environmentObject(container.purchaseManager)
                .environmentObject(container.adsManager)
                .environmentObject(container.userDataStore)
                .environmentObject(container.examHistoryStore)
                .environmentObject(container.studyViewModel)
                .environmentObject(container.mockExamViewModel)
                .environmentObject(container.settingsViewModel)
                .environmentObject(container.onboardingViewModel)
                .onReceive(container.purchaseManager.$hasRemoveAds) { _ in
                    container.refreshAdsState()
                }
        }
    }
}
