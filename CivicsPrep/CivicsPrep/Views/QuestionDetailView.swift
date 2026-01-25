import SwiftUI

struct QuestionDetailView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var userData: UserDataStore

    let question: Question

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(question.question.text(for: localization.language))
                    .font(.title2.bold())
                    .accessibilityLabel(Text(localization.localized("question_detail")))
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Text(localization.localized("answers"))
                        .font(.headline)
                    ForEach(question.answers, id: \.self) { answer in
                        Text("• \(answer.text(for: localization.language))")
                    }
                }
                if let explanation = question.explanation {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(localization.localized("explanation"))
                            .font(.headline)
                        Text(explanation.text(for: localization.language))
                    }
                }
                Divider()
                HStack(spacing: 12) {
                    Button {
                        userData.toggleBookmark(id: question.id)
                    } label: {
                        Label(userData.bookmarks.contains(question.id) ? localization.localized("remove_bookmark") : localization.localized("bookmark"), systemImage: "bookmark")
                    }
                    .buttonStyle(.bordered)

                    Button {
                        userData.toggleMastered(id: question.id)
                    } label: {
                        Label(userData.mastered.contains(question.id) ? localization.localized("remove_mastered") : localization.localized("mark_mastered"), systemImage: "checkmark.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
        }
        .navigationTitle(localization.localized("question_detail"))
    }
}
