import SwiftUI

struct ExamSetupView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var studyViewModel: StudyViewModel
    @EnvironmentObject private var mockExamViewModel: MockExamViewModel

    var body: some View {
        VStack(spacing: 20) {
            Text(localization.localized("choose_exam_length"))
                .font(.headline)
            Picker("", selection: $mockExamViewModel.examLength) {
                Text(localization.localized("exam_10")).tag(10)
                Text(localization.localized("exam_20")).tag(20)
            }
            .pickerStyle(.segmented)
            Text(localization.localized("exam_instructions"))
                .font(.footnote)
                .foregroundStyle(.secondary)
            Button {
                mockExamViewModel.startExam(from: studyViewModel.questions, length: mockExamViewModel.examLength)
            } label: {
                Text(localization.localized("start_exam"))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)
            Spacer()
        }
        .padding()
    }
}

struct ExamQuestionView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var mockExamViewModel: MockExamViewModel

    var body: some View {
        VStack(spacing: 20) {
            HStack {
                Text("\(localization.localized("exam_timer")): \(mockExamViewModel.elapsedSeconds)s")
                    .font(.footnote)
                Spacer()
                Text("\(mockExamViewModel.currentIndex + 1)/\(mockExamViewModel.sessionQuestions.count)")
                    .font(.footnote)
            }
            .foregroundStyle(.secondary)
            if let current = currentQuestion {
                Text(current.question.question.text(for: localization.language))
                    .font(.title2.bold())
                    .multilineTextAlignment(.leading)
                Spacer()
                Button {
                    mockExamViewModel.answerCurrent(correct: true)
                } label: {
                    Text(localization.localized("i_know"))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                Button {
                    mockExamViewModel.answerCurrent(correct: false)
                } label: {
                    Text(localization.localized("i_dont_know"))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
    }

    private var currentQuestion: ExamQuestionState? {
        guard mockExamViewModel.currentIndex < mockExamViewModel.sessionQuestions.count else {
            return nil
        }
        return mockExamViewModel.sessionQuestions[mockExamViewModel.currentIndex]
    }
}

struct ExamResultsView: View {
    @EnvironmentObject private var localization: LocalizationManager
    @EnvironmentObject private var mockExamViewModel: MockExamViewModel
    @EnvironmentObject private var examHistory: ExamHistoryStore

    var body: some View {
        VStack(spacing: 16) {
            Text(localization.localized("exam_results"))
                .font(.title.bold())
            Text("\(localization.localized("score")): \(mockExamViewModel.score()) / \(mockExamViewModel.sessionQuestions.count)")
                .font(.title2)
            if !mockExamViewModel.incorrectQuestions().isEmpty {
                List {
                    Section(localization.localized("review_incorrect")) {
                        ForEach(mockExamViewModel.incorrectQuestions()) { item in
                            Text(item.question.question.text(for: localization.language))
                                .lineLimit(2)
                        }
                    }
                }
            } else {
                Text("🎉")
                    .font(.largeTitle)
            }
            Button {
                let result = ExamResult(
                    id: UUID(),
                    date: Date(),
                    totalQuestions: mockExamViewModel.sessionQuestions.count,
                    correctCount: mockExamViewModel.score(),
                    incorrectQuestionIDs: mockExamViewModel.incorrectQuestions().map { $0.id }
                )
                examHistory.add(result)
                mockExamViewModel.reset()
            } label: {
                Text(localization.localized("start_exam"))
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)
        }
        .padding()
    }
}
