import SwiftUI

struct RootView: View {
    @EnvironmentObject private var onboarding: OnboardingViewModel

    var body: some View {
        Group {
            if onboarding.hasSeenOnboarding {
                MainTabView()
            } else {
                OnboardingView()
            }
        }
    }
}
