import SwiftUI

enum QuestionListFilter {
    case category(String)
    case bookmarks
    case mastered
}

struct QuestionListView: View {
    @EnvironmentObject private var studyViewModel: StudyViewModel
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var userData: UserDataStore

    let title: String
    let filter: QuestionListFilter

    var body: some View {
        List(filteredQuestions) { question in
            NavigationLink {
                QuestionDetailView(question: question)
            } label: {
                Text(question.question.text(for: localization.language))
                    .lineLimit(2)
            }
        }
        .navigationTitle(title)
        .searchable(text: $studyViewModel.searchText, prompt: localization.localized("search_placeholder"))
        .onAppear {
            studyViewModel.selectedCategory = selectedCategory
        }
        .onDisappear {
            studyViewModel.searchText = ""
        }
    }

    private var selectedCategory: String? {
        switch filter {
        case .category(let value):
            return value
        default:
            return nil
        }
    }

    private var filteredQuestions: [Question] {
        let base = studyViewModel.filteredQuestions(language: localization.language)
        switch filter {
        case .category:
            return base
        case .bookmarks:
            return base.filter { userData.bookmarks.contains($0.id) }
        case .mastered:
            return base.filter { userData.mastered.contains($0.id) }
        }
    }
}
