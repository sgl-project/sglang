import Foundation
import SwiftUI

final class AdsManager: ObservableObject {
    @Published var isEnabled: Bool

    init(isEnabled: Bool) {
        self.isEnabled = isEnabled
    }

    func updateEntitlement(removeAds: Bool) {
        isEnabled = !removeAds
    }
}

struct BannerAdPlaceholder: View {
    let text: String

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.2))
            Text(text)
                .font(.footnote)
                .foregroundColor(.secondary)
        }
        .frame(height: 60)
        .padding(.horizontal)
        .accessibilityLabel(Text(text))
    }
}
