import SwiftUI

struct StudyView: View {
    @EnvironmentObject private var studyViewModel: StudyViewModel
    @EnvironmentObject private var localization: LocalizationManager

    var body: some View {
        NavigationStack {
            List {
                Section(localization.localized("categories_title")) {
                    ForEach(studyViewModel.categories) { category in
                        NavigationLink(category.title) {
                            QuestionListView(title: category.title, filter: .category(category.title))
                        }
                    }
                }
                Section(localization.localized("bookmarks")) {
                    NavigationLink(localization.localized("bookmarks")) {
                        QuestionListView(title: localization.localized("bookmarks"), filter: .bookmarks)
                    }
                }
                Section(localization.localized("mastered")) {
                    NavigationLink(localization.localized("mastered")) {
                        QuestionListView(title: localization.localized("mastered"), filter: .mastered)
                    }
                }
            }
            .navigationTitle(localization.localized("study_title"))
            .safeAreaInset(edge: .bottom) {
                AdBannerView()
            }
        }
    }
}
