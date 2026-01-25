import Foundation

final class OnboardingViewModel: ObservableObject {
    @Published var hasSeenOnboarding: Bool {
        didSet {
            UserDefaults.standard.set(hasSeenOnboarding, forKey: StorageKeys.onboardingSeen)
        }
    }

    init() {
        hasSeenOnboarding = UserDefaults.standard.bool(forKey: StorageKeys.onboardingSeen)
    }
}
