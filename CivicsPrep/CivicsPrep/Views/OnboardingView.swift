import SwiftUI

struct OnboardingView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var onboarding: OnboardingViewModel

    var body: some View {
        VStack(spacing: 24) {
            Spacer()
            Image(systemName: "book.fill")
                .font(.system(size: 56))
                .foregroundStyle(.blue)
            Text(localization.localized("onboarding_title"))
                .font(.largeTitle.bold())
                .multilineTextAlignment(.center)
            Text(localization.localized("onboarding_body"))
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
            Spacer()
            Button {
                onboarding.hasSeenOnboarding = true
            } label: {
                Text(localization.localized("continue"))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
    }
}
