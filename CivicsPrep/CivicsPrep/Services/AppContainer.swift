import Foundation

@MainActor
final class AppContainer: ObservableObject {
    let localizationManager: LocalizationManager
    let purchaseManager: PurchaseManager
    let adsManager: AdsManager
    let userDataStore: UserDataStore
    let examHistoryStore: ExamHistoryStore
    let studyViewModel: StudyViewModel
    let mockExamViewModel: MockExamViewModel
    let settingsViewModel: SettingsViewModel
    let onboardingViewModel: OnboardingViewModel

    init() {
        localizationManager = LocalizationManager()
        purchaseManager = PurchaseManager()
        adsManager = AdsManager(isEnabled: true)
        userDataStore = UserDataStore()
        examHistoryStore = ExamHistoryStore()
        studyViewModel = StudyViewModel(repository: QuestionRepository())
        mockExamViewModel = MockExamViewModel()
        settingsViewModel = SettingsViewModel(purchaseManager: purchaseManager)
        onboardingViewModel = OnboardingViewModel()

        adsManager.updateEntitlement(removeAds: purchaseManager.hasRemoveAds)
    }

    func refreshAdsState() {
        adsManager.updateEntitlement(removeAds: purchaseManager.hasRemoveAds)
    }
}
