import Foundation

struct ExamQuestionState: Identifiable, Hashable {
    let id: String
    let question: Question
    var isCorrect: Bool?
}

@MainActor
final class MockExamViewModel: ObservableObject {
    @Published var examLength: Int = 10
    @Published private(set) var sessionQuestions: [ExamQuestionState] = []
    @Published private(set) var currentIndex: Int = 0
    @Published private(set) var isCompleted: Bool = false
    @Published private(set) var elapsedSeconds: Int = 0

    private var timer: Timer?

    func startExam(from questions: [Question], length: Int) {
        let count = min(length, questions.count)
        let selected = Array(questions.shuffled().prefix(count))
        sessionQuestions = selected.map { ExamQuestionState(id: $0.id, question: $0, isCorrect: nil) }
        currentIndex = 0
        isCompleted = false
        elapsedSeconds = 0
        startTimer()
    }

    func answerCurrent(correct: Bool) {
        guard currentIndex < sessionQuestions.count else { return }
        sessionQuestions[currentIndex].isCorrect = correct
        advance()
    }

    func advance() {
        if currentIndex + 1 < sessionQuestions.count {
            currentIndex += 1
        } else {
            finishExam()
        }
    }

    func finishExam() {
        isCompleted = true
        timer?.invalidate()
        timer = nil
    }

    func reset() {
        sessionQuestions = []
        currentIndex = 0
        isCompleted = false
        elapsedSeconds = 0
        timer?.invalidate()
        timer = nil
    }

    func score() -> Int {
        sessionQuestions.filter { $0.isCorrect == true }.count
    }

    func incorrectQuestions() -> [ExamQuestionState] {
        sessionQuestions.filter { $0.isCorrect == false }
    }

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self, !self.isCompleted else { return }
            self.elapsedSeconds += 1
        }
    }
}
