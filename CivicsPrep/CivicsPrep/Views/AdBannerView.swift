import SwiftUI

struct AdBannerView: View {
    @EnvironmentObject private var adsManager: AdsManager
    @EnvironmentObject private var localization: LocalizationManager

    var body: some View {
        if adsManager.isEnabled {
            BannerAdPlaceholder(text: localization.localized("ads_placeholder"))
        }
    }
}
